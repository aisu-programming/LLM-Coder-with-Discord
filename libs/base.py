# """ Libraries """
# import logging



# """ Loggers """
# BM_LOGGER = logging.getLogger("BaseModel")
# BM_LOGGER.setLevel(logging.INFO)
# BM_HANDLER = logging.StreamHandler()
# BM_HANDLER.setLevel(logging.INFO)
# BM_HANDLER.setFormatter(logging.Formatter(
#     "\n[%(levelname)s] (%(name)s) %(asctime)s | %(filename)s: %(funcName)s: %(lineno)03d | %(message)s",
#     datefmt="%m-%d %H:%M:%S"
# ))
# BM_LOGGER.addHandler(BM_HANDLER)



""" Classes """
class BaseModel(object):
    def __init__(self) -> None:
        self.stopping_sign = "User:"
        # self.SOU = "<|StartOfUser|>"  # Start Of User
        # self.EOU = "<|EndOfUser|>"    # End Of User
        # self.SOY = "<|StartOfYou|>"   # Start Of You
        # self.EOY = "<|EndOfYou|>"     # End Of You
    
    def __get_templated_message__(self, message):
        return f"User: {message}\nYou: "
    
    def __call__(self, message: str) -> str:
        # BM_LOGGER.info(f"message: {message}")
        msg_tpl = self.__get_templated_message__(message)
        # BM_LOGGER.info(f"msg_tpl:\n\n{msg_tpl}")
        response = self.generate_response(msg_tpl)
        # BM_LOGGER.info(f"Generated response:\n\n{response}")
        response = response.removeprefix(msg_tpl).removesuffix(self.stopping_sign)
        response = response.strip()
        # BM_LOGGER.info(f"Proccessed response:\n\n{response}")
        return response

    def generate_response(self, msg_tpl: str) -> str:
        raise NotImplementedError