def get_task(name):
    if name == "game24":
        from tot.tasks.game24 import Game24Task

        return Game24Task()
    elif name == "text":
        from tot.tasks.text import TextTask

        return TextTask()
    elif name == "crosswords":
        from tot.tasks.crosswords import MiniCrosswordsTask

        return MiniCrosswordsTask()
    elif name == "gsm":
        from tot.tasks.gsm8k import GSMTask

        return GSMTask()
    else:
        raise NotImplementedError
