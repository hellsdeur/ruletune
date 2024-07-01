from typing import List, Dict, Callable
import multiprocessing
import psutil
import time
import sys

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, hmine, association_rules


class ARM:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_basket(df: pd.DataFrame, purchase_identifier_column: str, product_identifier_column: str):
        df_source: pd.DataFrame = df.copy()
        tickets = {ticket: i for i, ticket in enumerate(sorted(list(df_source[purchase_identifier_column].unique())))}
        skus = {sku: i for i, sku in enumerate(sorted(list(df_source[product_identifier_column].unique())))}

        df_source["row"] = df_source[purchase_identifier_column].map(lambda x: tickets[x])
        df_source["col"] = df_source[product_identifier_column].map(lambda x: skus[x])
        df_source["data"] = np.ones((df_source.shape[0],), dtype=np.int8)

        sparse_matrix = csr_matrix(
            (df_source["data"], (df_source["row"], df_source["col"])),
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
    
    @staticmethod
    def save_basket(filename: str, basket: csr_matrix):
        np.savez(
            filename,
            data=basket.data,
            indices=basket.indices,
            indptr=basket.indptr,
            shape=basket.shape
        )

    @staticmethod
    def load_basket(filename):
        loader = np.load(filename)
        return csr_matrix(
            (loader['data'], loader['indices'], loader['indptr']),
            shape=loader['shape']
        )
    
    @staticmethod
    def algorithm_wrapper(queue: multiprocessing.Queue, algorithm: Callable, basket: pd.DataFrame, min_support: float = 0.1) -> pd.DataFrame:
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

        arrays = []

        for i in range(a, b):
            arrays.append(np.linspace(1*10**i, 9*10**i, 9))

        return np.concatenate(arrays)

    @staticmethod
    def search_rules(
        basket: pd.DataFrame, algorithms: List[Callable] = [apriori, fpgrowth, fpmax, hmine],
        etime: int = 3600, df_sku: pd.DataFrame = None, sku_col: str = None, description_col: str = None,
        save_directory: str = "./"
    ):
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
                    basket=basket,
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
        df_summary.to_csv(
            f"{save_directory}/summary.csv",
            index=False
        )
        return df_summary
