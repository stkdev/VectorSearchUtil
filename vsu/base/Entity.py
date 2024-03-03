from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, MetaData, ForeignKey, JSON, Float


class Base(DeclarativeBase):
    pass

class J_InfoVector(Base):
    __tablename__ = 'j_info_vector'

    id = Column(Integer, primary_key=True)
    pk = Column(Integer)
    vector_id = Column(Integer)

    def get_column(self):
        return ["id", "pk", "vector_id"]

    def get_list(self):
        return [self.id, self.pk, self.vector_id]

    def __repr__(self):
        return "<j_info_vector(id={}, pk={}, vector_id={})".format(self.id, self.pk, self.vector_id)

class T_Info(Base):
    __tablename__ = 'info'

    id = Column(Integer, primary_key=True)
    target = Column(String)
    pk = Column(String)
    option1 = Column(String)
    option2 = Column(String)
    option3 = Column(String)
    option4 = Column(String)
    option5 = Column(String)

    def get_column(self):
        return ["id", "target", "pk", "option1", "option2", "option3", "option4", "option5"]

    def get_list(self):
        return [self.id, self.target, self.pk,
                self.option1, self.option2, self.option3, self.option4, self.option5]

    def __repr__(self):
        return "<info(id={}, target={}, pk={}, op1={}, op2={}, op3={}, op4={}, op5={})".format(self.id, self.target, self.pk, self.option1, self.option2, self.option3, self.option4, self.option5)

class T_Vector(Base):
    __tablename__ = 'vector'

    id = Column(Integer, primary_key=True)
    target = Column(String)
    pk = Column(String)
    vector = Column(JSON)

    def get_column(self):
        return ["id", "target", "pk", "vector"]

    def get_list(self):
        return [self.id, self.target, self.pk,self.vector]

    def __repr__(self):
        return "<vector(id={}, target={}, pk={}, vector={})".format(self.id, self.target, self.pk ,self.vector)

