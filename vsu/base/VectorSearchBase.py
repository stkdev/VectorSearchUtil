from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import sqlite3
import sqlite_vss
import pandas as pd
import numpy as np
from typing import List


sqlite3.register_adapter(list, lambda l: ';'.join([str(i) for i in l]))
sqlite3.register_converter('List', lambda s: [float(item) for item in s.split(bytes(b';'))])


class VectorSearchBase:

    def __init__(self, db_name=':memory:'):

        self.data = None
        self.save_columns = ["target", "option1", "option2", "option3", "option4","option5"]

        self.zeroshot_labels = None
        self.zeroshot_vec = None

        self.db = sqlite3.connect(db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.db.enable_load_extension(True)

        self.config = {}

        sqlite_vss.load(self.db)

        self.set_config()
        self.init_model()
        self.init_db()


    def set_config(self, **kwargs):
        self.config = kwargs
        return

    # override
    def init_model(self):
        self.tokenizer = None
        self.model = None
        self.preprocess = None

        self.vec_size = None

    def init_db(self):
        vec_size = self.vec_size

        sql = """
        create table if not exists data (
          id integer primary key,
          pk text,
          %s,
          vector List
        );
        """ % (",".join([" ".join([c, "text"]) for c in self.save_columns]),)
        self.db.execute(sql)

        sql = f"""
        CREATE VIRTUAL TABLE if not exists vss USING vss0 (
            vector({vec_size})
        );
        """
        self.db.execute(sql)

        if self.data is None:
            self.__set_data4db()

    def __set_data4db(self):
        sql = f"""
        select %s from data;
        """ % (",".join(["data."+c for c in ["pk"]+self.save_columns+["vector"]]),)

        tmp = pd.DataFrame(self.db.execute(sql).fetchall(), columns=["pk"]+self.save_columns+["vector"])
        if 0 < tmp.shape[0]:
            self.data = tmp
        return

    def __serialize(self, vector: List[float]) -> bytes:
        return np.asarray(vector).astype(np.float32).tobytes()

    def insert_data(self, row):
        with self.db:
            target_data = self.db.execute('''
        select * from data where pk = ?
      ''', (row["pk"],)).fetchone()

            if target_data is None:
                self.db.execute(f'''
            INSERT INTO data(pk,{",".join(self.save_columns)},vector)
            VALUES ({",".join(["?"]*(len(self.save_columns)+2))})
        ''', tuple(row[["pk"]+self.save_columns+["vector"]]))

                last_id = self.db.execute('SELECT last_insert_rowid()').fetchone()[0]

                self.db.execute('''
            INSERT INTO vss(rowid, vector)
            VALUES (?, ?)
        ''', (last_id, self.__serialize(row["vector"])))

    def reset_db(self):
        self.db.execute("DROP TABLE data;")
        self.db.execute("DROP TABLE vss;")
        self.data = None

    def set_data(self, data, append=False, sp=10):
        data = data.copy()

        if 'target' not in data.columns:
            return

        if append:
            pass
        elif self.data is not None:
            self.reset_db()
            self.init_db()

        prefix = self.config.get('query_prefix', '')

        data["label"] = [prefix+t for t in data["target"].tolist()]
        data["pk"] = data["label"]

        if "vector" not in data.columns:
            data["vector"] = self.__trans_vec_main(data["label"].to_list(), sp=sp, verbose=True)
            print("add vector")

        for c in self.save_columns:
            if c not in data.columns:
                data[c] = None

        if append and self.data is not None:
            self.data = pd.concat([self.data, data[["pk"]+self.save_columns+["vector"]]]).drop_duplicates(subset='pk')
        else:
            self.data = data[["pk"]+self.save_columns+["vector"]]

        for i, row in data.iterrows():
            self.insert_data(row)

        self.__set_data4db()

        return

    def set_zeroshot_labels(self, arr):
        self.zeroshot_labels = arr
        self.zeroshot_vec = self.__trans_vec_sub(arr)

        return

    # override
    def do_zeroshot(self):
        pass

    # override
    def _trans_vec_main_func(self, ar):
        pass

    # override
    def _trans_vec_sub_func(self, ar):
        pass

    def __trans_vec_main(self, ary, sp=10, verbose=False):

        l_max = len(ary)
        ret = []
        for i in range(0, (1+l_max // sp)):
            if i*sp == l_max:
                break

            ar = ary[(i*sp):(min(sp*(i+1), l_max))]
            features = self._trans_vec_main_func(ar)

            ret.extend(features)
            if verbose:
                print(f'{i*sp} / {l_max}')
        if verbose:
            print("complete")

        return ret

    def __trans_vec_sub(self, ary, sp=10, verbose=False):
        """

        """
        l_max = len(ary)
        ret = []
        for i in range(0, (1+l_max // sp)):
            if i*sp == l_max:
                break

            ar = ary[(i*sp):(min(sp*(i+1), l_max))]

            v = self._trans_vec_sub_func(ar)
            ret.extend(v)
            if verbose:
                print(f'{i*sp} / {l_max}')
        if verbose:
            print("complete")

        return ret

    def __search_similar_embeddings(self, query_embedding, k=5):
        results = self.db.execute(f'''
        SELECT data.id,{",".join(["data."+c for c in self.save_columns])}, vss.distance
        FROM vss
        JOIN data ON vss.rowid = data.id
        WHERE vss_search(vss.vector, vss_search_params(?, ?))
        ORDER BY vss.distance
        LIMIT ?
    ''', (self.__serialize(query_embedding), k, k))
        return results.fetchall()

    def __q(self, q):
        embeddings = self.__trans_vec_main([q], sp=10)
        return embeddings[0]

    def query(self, query, k=5):
        if self.data is None or self.data.shape[0] == 0:
            return

        qry = self.__q(query)
        return self.__search_similar_embeddings(qry, k=k)

    def query_with_info(self, query, k=5):
        result = self.query(query, k=k)
        return pd.DataFrame(result, columns=["id"]+self.save_columns+["distance"]).dropna(how='all', axis=1)

    def MLP_Classifier(self, y_label, skip_build=False, hidden_layer_sizes=(100,)):
        if self.data is None:
            return

        y = self.data[y_label]
        X = pd.DataFrame(self.data["vector"].tolist())
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=810)

        if skip_build and self.clf is not None:
            clf = self.clf
        else:
            clf = MLPClassifier(random_state=810, max_iter=300, hidden_layer_sizes=hidden_layer_sizes, verbose=False).fit(X_train, y_train)
            self.clf = clf

        print("score:", clf.score(X_test, y_test))
        return clf.predict(X), clf.predict_proba(X)

    def MLP_Regressor(self, y_label, skip_build=False, hidden_layer_sizes=(100,)):
        if self.data is None:
            return

        y = self.data[y_label]
        X = pd.DataFrame(self.data["vector"].tolist())
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=810)

        if skip_build and self.regr is not None:
            regr = self.regr
        else:
            regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=hidden_layer_sizes, verbose=False).fit(X_train, y_train)
            self.regr = regr

        print("score:", regr.score(X_test, y_test))
        print("corr:", np.corrcoef(y_test, regr.predict(X_test)))

        self.X_test = X_test
        self.y_test = y_test

        return regr.predict(X)
