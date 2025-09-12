


class GlobalAgentState:

    _instance = None
    
    def __new__(cls):
        if cls._instance is None:  # first time -> create instance
            cls._instance = super(GlobalAgentState, cls).__new__(cls)
            cls._instance.chatAgent = None
            cls._instance.responseAgent = None
        return cls._instance
