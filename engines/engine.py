
class BaseEngine():

    def __init__(self,name):
        self.name = name

    def execute(self,input,debug = False):
        self.update_engine_input(input)
        return self.execute_engine(debug = debug)



    def get_output(self,direction):
        return self.get_engine_output(direction)

    def update_engine_input(self,input):
        raise NotImplementedError

    def execute_engine(self,debug=False):
        raise NotImplementedError

    def get_engine_output(self,direction):
        raise NotImplementedError



