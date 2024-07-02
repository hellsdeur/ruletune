from typing import List, Dict, Callable
import multiprocessing
import psutil
import time
import sys
import re

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, hmine, association_rules


class ARM:
    def __init__(self, df: pd.DataFrame, transaction_id_column: str, product_id_column) -> None:
        """
        Initialize the ARM (Association Rule Mining) class.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.
            transaction_id_column (str): The name of the column containing transaction IDs.
            product_id_column (str): The name of the column containing product IDs.
        """
        self.df: pd.DataFrame = df
        self.tid_column: str = transaction_id_column
        self.pid_column: str = product_id_column
        self.basket: pd.DataFrame = self.create_basket()

    def create_basket(self) -> pd.DataFrame:
        """
        Create a basket format DataFrame from the transaction data.

        This method transforms the transaction data into a sparse matrix where each row represents a transaction
        and each column represents a product. The matrix is then converted into a DataFrame in basket format,
        where the presence of a product in a transaction is indicated by a boolean value.

        Returns:
            pd.DataFrame: A DataFrame in basket format with transactions as rows and products as columns.
        """

        tickets = {ticket: i for i, ticket in enumerate(sorted(list(self.df[self.tid_column].unique())))}
        skus = {sku: i for i, sku in enumerate(sorted(list(self.df[self.pid_column].unique())))}

        self.df["row"] = self.df[self.tid_column].map(lambda x: tickets[x])
        self.df["col"] = self.df[self.pid_column].map(lambda x: skus[x])
        self.df["data"] = np.ones((self.df.shape[0],), dtype=np.int8)

        sparse_matrix = csr_matrix(
            (self.df["data"], (self.df["row"], self.df["col"])),
            shape=(len(tickets), len(skus)),
            dtype=np.int8
        )
        sparse_matrix = np.nan_to_num(sparse_matrix, copy=False)
        sparse_matrix = sparse_matrix.astype("bool")
        
        basket = pd.DataFrame.sparse.from_spmatrix(
            sparse_matrix,
            index=tickets,
            columns=skus,
        )

        return basket
    
    def save_basket(self, filename: str):
        """
        Save the basket DataFrame to a file in compressed NumPy .npz format.

        Args:
            filename (str): The name of the file to save the basket data.
        """

        np.savez(
            filename,
            data=self.basket.data,
            indices=self.basket.indices,
            indptr=self.basket.indptr,
            shape=self.basket.shape
        )

    @staticmethod
    def load_basket(filename):
        """
        Load a basket from a compressed NumPy .npz file.

        Args:
            filename (str): The name of the file to load the basket data from.

        Returns:
            csr_matrix: The basket data as a sparse CSR matrix.
        """

        loader = np.load(filename)
        return csr_matrix(
            (loader['data'], loader['indices'], loader['indptr']),
            shape=loader['shape']
        )
    
    @staticmethod
    def algorithm_wrapper(queue: multiprocessing.Queue, algorithm: Callable, basket: pd.DataFrame, min_support: float = 0.1) -> pd.DataFrame:
        """
        Execute a given association rule mining algorithm on the basket data.

        Args:
            queue (multiprocessing.Queue): The multiprocessing queue to put the result into.
            algorithm (Callable): The association rule mining algorithm to apply.
            basket (pd.DataFrame): The basket data on which to run the algorithm.
            min_support (float, optional): The minimum support threshold for the algorithm. Defaults to 0.1.

        Returns:
            pd.DataFrame: The resulting DataFrame containing the support and itemsets,
                        or an empty DataFrame with columns ["support", "itemsets"] in case of errors.
        """

        try:
            result = algorithm(basket, min_support=min_support, use_colnames=True)
        except ValueError as ve:
            result = pd.DataFrame(columns=["support", "itemsets"])
        except np.core._exceptions._ArrayMemoryError as ame:
            result = pd.DataFrame(columns=["support", "itemsets"])
        queue.put(result)
        return result
    
    @staticmethod
    def get_frequent_itemsets(algorithm: Callable, basket: pd.DataFrame, min_support: float = 0.1, etime: int = 3600) -> pd.DataFrame:
        """
        Execute a given association rule mining algorithm to find frequent itemsets from the basket data
        within a specified time limit or memory threshold.

        Args:
            algorithm (Callable): The association rule mining algorithm to apply.
            basket (pd.DataFrame): The basket data on which to run the algorithm.
            min_support (float, optional): The minimum support threshold for the algorithm. Defaults to 0.1.
            etime (int, optional): The maximum elapsed time (in seconds) before terminating the algorithm.
                                Defaults to 3600 seconds (1 hour).

        Returns:
            pd.DataFrame: The DataFrame containing the frequent itemsets with columns ["support", "itemsets"],
                        or None if no itemsets were found within the given constraints.
        """
            
        result_queue = multiprocessing.Queue()
        algorithm_process = multiprocessing.Process(target=ARM.algorithm_wrapper, args=(result_queue, algorithm, basket, min_support))
        algorithm_process.start()
        start_time: float = time.time()

        while algorithm_process.is_alive():
            if (psutil.virtual_memory().percent >= 90) or (time.time() - start_time > etime):
                algorithm_process.terminate()
                algorithm_process.join()
                break

        algorithm_process.join()

        if not result_queue.empty():
            frequent_itemsets = result_queue.get()
        else:
            frequent_itemsets = None

        return frequent_itemsets

    @staticmethod
    def generate_rules(frequent_itemsets: pd.DataFrame, metric: str = "lift", min_threshold: float = 0.001) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets using the specified metric and minimum threshold.

        Args:
            frequent_itemsets (pd.DataFrame): The DataFrame containing frequent itemsets with columns ["support", "itemsets"].
            metric (str, optional): The metric to use for rule generation, e.g., "lift", "confidence". Defaults to "lift".
            min_threshold (float, optional): The minimum threshold for the specified metric. Defaults to 0.001.

        Returns:
            pd.DataFrame: The DataFrame containing the generated association rules with columns:
                        ["antecedents", "consequents", "antecedent support", "consequent support",
                        "support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"].
                        Returns an empty DataFrame if no rules were generated.
        """

        try:
            rules = association_rules(
                df=frequent_itemsets,
                support_only=False,
                metric=metric,
                min_threshold=min_threshold
            )
            if rules.shape[0] == 0:
                rules = association_rules(
                    df=frequent_itemsets,
                    support_only=True,
                    metric=metric,
                    min_threshold=min_threshold
                )
        except KeyError as ke:
            rules = association_rules(
                df=frequent_itemsets,
                support_only=True,
                metric=metric,
                min_threshold=min_threshold
            )
        
        if rules.shape[0] == 0:

            rules = pd.DataFrame(columns=[
                "antecedents", "consequents", "antecedent support", "consequent support",
                "support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"
            ])

        rules = rules.sort_values(by="confidence", ascending=False)

        return rules
    
    @staticmethod
    def describe_rules(rules: pd.DataFrame, df_sku: pd.DataFrame, sku_col: str, description_col: str) -> pd.DataFrame:
        """
        Convert SKU identifiers in association rules to their corresponding descriptions.

        Args:
            rules (pd.DataFrame): DataFrame containing association rules with columns ["antecedents", "consequents"].
            df_sku (pd.DataFrame): DataFrame mapping SKU identifiers to descriptions with columns [sku_col, description_col].
            sku_col (str): Name of the column in df_sku containing SKU identifiers.
            description_col (str): Name of the column in df_sku containing SKU descriptions.

        Returns:
            pd.DataFrame: DataFrame with association rules where SKU identifiers in "antecedents" and "consequents"
                        columns are replaced with their corresponding descriptions from df_sku.
        """
        def sku_to_description(x: frozenset, df_sku: pd.DataFrame, description_col: str) -> frozenset:
            result = []
            for element in x:
                try:
                    result.append(df_sku.loc[element, description_col])
                except KeyError as e:
                    result.append(element)
            return frozenset(result)

        df_sku = df_sku.set_index(sku_col)
        rules.iloc[:, :2] = rules.iloc[:, :2].map(lambda x: sku_to_description(x, df_sku, description_col))
        return rules
    
    @staticmethod
    def get_min_support_values(a: int = -3, b: int = 0):
        """
        Generate an array of minimum support values for association rule mining.

        Args:
            a (int, optional): Exponent of the lower bound of minimum support values (10**a). Defaults to -3.
            b (int, optional): Exponent of the upper bound of minimum support values (10**b). Defaults to 0.

        Returns:
            np.ndarray: Array of minimum support values logarithmically spaced between 10**a and 10**b.
        """

        arrays = []

        for i in range(a, b):
            arrays.append(np.linspace(1*10**i, 9*10**i, 9))

        return np.concatenate(arrays)

    def search_rules(
        self, algorithms: List[Callable] = [apriori, fpgrowth, fpmax, hmine],
        etime: int = 3600, df_sku: pd.DataFrame = None, sku_col: str = None, description_col: str = None,
        save_directory: str = None
    ) -> pd.DataFrame:
        """
        Search for association rules using multiple algorithms and different minimum support values.

        This method iterates over a list of association rule mining algorithms and searches for frequent itemsets
        using a range of minimum support values. For each algorithm and minimum support value combination,
        it generates association rules and collects summary statistics.

        Args:
            algorithms (List[Callable], optional): List of association rule mining algorithms to use.
                                                Defaults to [apriori, fpgrowth, fpmax, hmine].
            etime (int, optional): Maximum elapsed time (in seconds) before terminating each algorithm.
                                Defaults to 3600 seconds (1 hour).
            df_sku (pd.DataFrame, optional): DataFrame mapping SKU identifiers to descriptions.
                                            Used for describing rules if provided. Defaults to None.
            sku_col (str, optional): Name of the column in df_sku containing SKU identifiers.
                                    Required if df_sku is provided. Defaults to None.
            description_col (str, optional): Name of the column in df_sku containing SKU descriptions.
                                            Required if df_sku is provided. Defaults to None.
            save_directory (str, optional): Directory path to save intermediate and final results (CSV files).
                                            Defaults to None.

        Returns:
            pd.DataFrame: DataFrame summarizing the results of the association rule mining search with columns:
                        ["algorithm", "min_support", "num_freq_itemsets", "num_rules", "max_confidence", "notes"].
                        Each row corresponds to the results for a specific algorithm and minimum support value combination.
        """

        # dictionary that holds data for the final dataframe summary
        results: Dict = {
            "algorithm": [],
            "min_support": [],
            "num_freq_itemsets": [],
            "num_rules": [],
            "max_confidence": [],
            "notes": []
        }

        # list of values of min_support to search
        min_supports = ARM.get_min_support_values()

        for algorithm in algorithms:

            for min_support in min_supports:
                # frequent_itemsets is either None
                # or a DataFrame with 0 rows,
                # or a DataFrame with more than 0 lines

                sys.stdout.write(f"\rSearching min_support for {algorithm.__name__}: {min_support}")
                sys.stdout.flush()

                frequent_itemsets = ARM.get_frequent_itemsets(
                    algorithm=algorithm,
                    basket=self.basket,
                    min_support=min_support,
                    etime=etime
                )

                if isinstance(frequent_itemsets, pd.DataFrame) and frequent_itemsets.shape[0] > 0:
                    break

            # memory threshold exceeded
            if frequent_itemsets is None:
                num_freq_itemsets = None
                num_rules = None
                max_confidence = None
                note = "Memory threshold exceeded."
            
            # expected behavior
            elif frequent_itemsets.shape[0] > 0:

                num_freq_itemsets = frequent_itemsets.shape[0]

                rules = ARM.generate_rules(frequent_itemsets=frequent_itemsets)

                if rules.shape[0] > 0:
                    if isinstance(df_sku, pd.DataFrame):
                        rules = ARM.describe_rules(
                            rules=rules,
                            df_sku=df_sku,
                            sku_col=sku_col,
                            description_col=description_col
                        )
                    max_confidence = rules["confidence"].sort_values(ascending=False).iloc[0]
                    num_rules = rules.shape[0]
                    note = ""
                else:
                    max_confidence = None
                    num_rules = 0
                    note = "No rules could be found in itemsets."
                
                if save_directory:
                    frequent_itemsets.to_csv(
                        f"{save_directory}/fqits_{algorithm.__name__}_{min_support}.csv",
                        index=False
                    )

                    rules.to_csv(
                        f"{save_directory}/rules_{algorithm.__name__}_{min_support}.csv",
                        index=False
                    )

            elif frequent_itemsets.shape[0] == 0:
                num_freq_itemsets = 0
                num_rules = 0
                max_confidence = None
                note = "No frequent itemsets found."

            results["algorithm"].append(algorithm.__name__)
            results["min_support"].append(min_support)
            results["num_freq_itemsets"].append(num_freq_itemsets)
            results["num_rules"].append(num_rules)
            results["max_confidence"].append(max_confidence)
            results["notes"].append(note)

            print("")

        df_summary: pd.DataFrame = pd.DataFrame(results)

        if save_directory:
            df_summary.to_csv(
                f"{save_directory}/summary.csv",
                index=False
            )

        return df_summary
    

def read_dat(path: str, sep: str = ' ') -> pd.DataFrame:
    with open(path, "r") as f:
        lines = f.readlines()
    
    data = {
        "tid": [],
        "item": []
    }
    num_lines = len(lines)
    zeros_len = len(str(num_lines))

    for i, line in enumerate(lines):
        tid = str(i).zfill(zeros_len)
        for item in line.split(sep):
            if re.search(r"\d+", item):
                data["tid"].append(tid)
                data["item"].append(item)

    return pd.DataFrame(data)


if __name__ == "__main__":

    df = read_dat("./examples/brijs/retail.dat")
    print(df)

    arm = ARM(
        df=df,
        transaction_id_column="tid",
        product_id_column="item"
    )

    df_summary = arm.search_rules()
    print(df_summary)
