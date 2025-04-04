from base import Base

class EEG(Base):
    
    def __init__(self, filepath:str):
        super().__init__()
    
    pass

eeg = EEG("hello")
eeg.test_fun()