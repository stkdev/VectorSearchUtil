from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

from sqlalchemy import select

import sqlite3
# import sqlite_vss
import pandas as pd
import numpy as np
from typing import List

from vsu.base.Entity import Base
from vsu.base.Entity import T_Info, T_Vector, J_InfoVector

from voyager import Index, Space

import os

# sqlite3.register_adapter(list, lambda l: ';'.join([str(i) for i in l]))
# sqlite3.register_converter('List', lambda s: [float(item) for item in s.split(bytes(b';'))])


class VectorSearchBase:

    def __init__(self, save_name=None, echo=False):

        self.data = None
        self.info = None
        self.vector = None
        self.save_columns = ["target", "option1", "option2", "option3", "option4","option5"]

        self.zeroshot_labels = None
        self.zeroshot_vec = None

        self.Base = Base()
        self.save_name = save_name
        self.db_name = '' if save_name is None else f'/{save_name}.db'
        self.voy_name = None if save_name is None else f'{save_name}.voy'

        self.engine = create_engine(f'sqlite://{self.db_name}', echo=echo)
        self.session = Session(self.engine)

        self.index = None

        # self.db = sqlite3.connect(db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        # self.db.enable_load_extension(True)

        self.config = {}

        # sqlite_vss.load(self.db)

        # self.set_config()
        self.init_model()
        self.init_db()
        self.init_voyager()


    def set_config(self, **kwargs):
        self.config = kwargs
        return

    # override
    def init_model(self):
        self.tokenizer = None
        self.model = None
        self.preprocess = None

        self.vec_size = None

    def init_db(self, drop:bool = False):
        vec_size = self.vec_size

        if drop:
            self.Base.metadata.drop_all(self.engine)

        if False:
            self.Base = automap_base()
            self.Base.prepare(self.engine, reflect=True)

        self.Base.metadata.create_all(self.engine)

        self.__set_data4db()
        # self.data = self.session.query(T_Info)

        # if self.data is None:
        #     self.__set_data4db()

    def init_voyager(self):
        self.index = Index(Space.Cosine, num_dimensions=self.vec_size)

        if self.voy_name is not None and os.path.isfile(self.voy_name):
            self.index = self.index.load(self.voy_name)

        return

    def __set_data4db(self):
        info = self.session.query(T_Info).all()
        vector = self.session.query(T_Vector).all()
        j_info_vector = self.session.query(J_InfoVector).all()

        if 0 < len(info):
            self.info = pd.DataFrame([o.get_list() for o in info], columns=info[0].get_column())
            self.vector = pd.DataFrame([v.get_list() for v in vector], columns=vector[0].get_column())
            self.j_info_vector = pd.DataFrame([j.get_list() for j in j_info_vector], columns=j_info_vector[0].get_column())

            self.data = pd.merge(self.info, self.vector[["pk", "vector"]], on="pk", how="left")

        # sql = f"""
        # select %s from data;
        # """ % (",".join(["data."+c for c in ["pk"]+self.save_columns+["vector"]]),)
        #
        # tmp = pd.DataFrame(self.db.execute(sql).fetchall(), columns=["pk"]+self.save_columns+["vector"])
        # if 0 < tmp.shape[0]:
        #     self.data = tmp
        return

    def __serialize(self, vector: List[float]) -> bytes:
        return np.asarray(vector).astype(np.float32)#.tobytes()

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
            self.Base.metadata.drop_all(self.engine)
            self.Base.metadata.create_all(self.engine)
            if self.voy_name is not None and os.path.isfile(self.voy_name):
                # os.remove(self.voy_name)
                self.init_voyager()

            # self.reset_db()
            # self.init_db()

        prefix = self.config.get('query_prefix', '')

        data["pk"] = [prefix+t for t in data["target"].tolist()]
        # data["pk"] = data["label"]


        # if "vector" not in data.columns:
        #     data["vector"] = self.__trans_vec_main(data["label"].to_list(), sp=sp, verbose=True)
        #     print("add vector")

        vec_target = data[~data.duplicated(subset='pk')]

        vec_target["vector"] = self.__trans_vec_main(vec_target["pk"].to_list(), sp=sp, verbose=True)
        # vectors32 = np.stack(vec_target["vector"].to_numpy()).astype(np.float32)

        # add info
        for c in self.save_columns:
            if c not in data.columns:
                data[c] = None

        dat = []
        for i,d in data.iterrows():
            add = T_Info(
                target = d["target"],
                pk = d["pk"],
                option1 = d["option1"],
                option2 = d["option2"],
                option3 = d["option3"],
                option4 = d["option4"],
                option5 = d["option5"],
            )
            dat.append(add)

        self.session.add_all(dat)
        self.session.commit()

        # add vector
        dat = []
        for i,d in vec_target.iterrows():
            add = T_Vector(
                target = d["target"],
                pk = d["pk"],
                vector = d["vector"],
            )
            dat.append(add)

        self.session.add_all(dat)
        self.session.commit()

        # add vector to voyager
        if 0 < vec_target.shape[0]:
            vectors32 = np.stack(vec_target["vector"].to_numpy()).astype(np.float32)

            ids = self.index.add_items(vectors32)

            if self.voy_name is not None:
                self.index.save(self.voy_name)

            # add j
            dat =[]
            for pk, id in zip(vec_target["pk"].tolist(), ids):
                add = J_InfoVector(
                    pk=pk,
                    vector_id=id,
                )
                dat.append(add)

            self.session.add_all(dat)
            self.session.commit()


        # if append and self.data is not None:
        #     self.data = pd.concat([self.data, data[["pk"]+self.save_columns+["vector"]]]).drop_duplicates(subset='pk')
        # else:
        #     self.data = data[["pk"]+self.save_columns+["vector"]]


        # for i, row in data.iterrows():
        #     self.insert_data(row)
        #
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
        q = self.__serialize(query_embedding)

        neighbors, distances = self.index.query(self.__serialize(query_embedding), k=k)

    #     results = self.db.execute(f'''
    #     SELECT data.id,{",".join(["data."+c for c in self.save_columns])}, vss.distance
    #     FROM vss
    #     JOIN data ON vss.rowid = data.id
    #     WHERE vss_search(vss.vector, vss_search_params(?, ?))
    #     ORDER BY vss.distance
    #     LIMIT ?
    # ''', (self.__serialize(query_embedding), k, k))

        return neighbors, distances

    def __q(self, q):
        embeddings = self.__trans_vec_main([q], sp=10)
        return embeddings[0]

    def query(self, query, k=5):
        if self.info is None or (self.info.shape[0] == 0):
            return pd.DataFrame([])

        qry = self.__q(query)
        neighbors, distances = self.__search_similar_embeddings(qry, k=k)
        result = pd.DataFrame({
            "vector_id": neighbors,
            "distance": distances,
        })
        return result

    def query_with_info(self, query, k=5):
        result = self.query(query, k=k)

        if result is None or (result.shape[0] == 0):
            return pd.DataFrame([])

        tmp = pd.merge(
            result,
            self.j_info_vector,
            on="vector_id",
            how="left",
        )
        result = pd.merge(
            tmp,
            self.info,
            on="pk",
            how="left",
        ).dropna(how='all', axis=1)

        return result
        # return pd.DataFrame(result, columns=["id"]+self.save_columns+["distance"]).dropna(how='all', axis=1)

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
