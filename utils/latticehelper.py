from utils.consts import Consts
import random


class LatticeHelper():
    def __init__(self, lattice_template, lattice_error_template, lattice_template_out="Mebt_template_out.dat",
                 max_error=0):
        '''
        TODO:You should creat a template for lattice.dat, and the keywords to replace need to be added in conts.py(Conts),
         and enable the keywords in enabled_keywords.
        关键字需要添加到conts.py文件中，并且在enabled_keywords中开启可以使用的关键字
        example：
            FIELD_MAP  0070 380 0 26 1QMAGNET 0 0 0 q120
            SUPERPOSE_MAP 0 0 0    0 0 0
            FIELD_MAP  0070 380 0 26 1DMAGNET 0 0 0 QL120X
            SUPERPOSE_MAP 0 0 0    0 0 0
            FIELD_MAP  0070 380 0 26 2DMAGNET 0 0 0 QL120Y
            SUPERPOSE_MAP 205.05 0 0    0 0 0
            MATCH_FAM_GRAD 12 0
            FIELD_MAP  0070 400 0 26 2QMAGNET 0 0 0 q150
        :param lattice_template: 这个模板文件中的关键字必须要和下面定义的关键字
        :param lattice_out:  模板文件的输出目录
        '''
        self.lattice_template = lattice_template
        self.enabled_keywords = [Consts.D_MAGENET_X, Consts.D_MAGENET_Y, Consts.SOL]
        self.lattice_template_out = lattice_template_out
        self.lattice_error_template = lattice_error_template
        self.random_error_num = random.random()
        self.max_error = max_error
        self.error_element_num = 7  # TODO：这里的9包含了MEBT段的1个Q铁的误差
        self.MEBT_out_error_rate = 0.2
        # self.error_list_x = [(-1 + 2 * random.random())*max_error for i in range(self.error_element_num)]
        # self.error_list_y = [(-1 + 2 * random.random())*max_error for i in range(self.error_element_num)]
        self.error_list_x = []
        self.error_list_y = []
        for i in range(self.error_element_num):
            error_x = (-1 + 2 * random.random()) * max_error
            error_y = (-1 + 2 * random.random()) * max_error
            if i < self.error_element_num - 1:  # 给超导段添加的误差
                self.error_list_x.append(error_x)
                self.error_list_y.append(error_y)
            else:  # 给MEBT添加的误差
                self.error_list_x.append(error_x * self.MEBT_out_error_rate)
                self.error_list_y.append(error_y * self.MEBT_out_error_rate)
        print()
        # random.seed(100)

    def check_keys(func):
        def wrapper(ctx, value_dict, *args, **kwargs):
            keys = value_dict.keys()
            for key in keys:
                if key not in ctx.enabled_keywords:
                    raise Exception("Error", "需要操作的key不在LatticeHelper允许的字典中")
            return func(ctx, value_dict, *args, **kwargs)

        return wrapper

    @check_keys
    def generate_lattice_template(self, fixed_params_dict, error_mode=False, error_element=[], error_rate=0.01):
        print(f"fixed_params_dict = {fixed_params_dict}")
        '''
        :param fixed_params_dict: 需要写入固定值的参数列表
        :param error_mode: 是否开启误差模式
        :param error_element: 开启误差的元件
        :param error_rate: 误差范围-1*error_rate ->  1*error_rate
        :return:
        '''
        # print(f"error_mode = {error_mode}")
        self.random_error_num = random.random()
        print(f"Before add error:SOL {fixed_params_dict}")
        if error_mode:
            if len(error_element) == 0:
                raise Exception("开启了误差模式，没有给出需要添加误差的元件")
            for element in error_element:
                if element not in fixed_params_dict:
                    raise KeyError("需要添加误差的原件不在fixed_params_dict中")
                else:
                    org_value = fixed_params_dict[element]
                    err_value = []
                    for item in org_value:
                        delta = (-1 + 2 * self.random_error_num) * error_rate * item  # -error_rate   ->   error_rate
                        new_value = round(item + delta, 3)

                        # TODO: 下面这一段是为了超导段的螺线管来限制最大最小值的
                        if new_value > 1:
                            new_value = 1
                        if new_value < -1:
                            new_value = -1
                        # TODO: 上面这一段是为了超导段的螺线管来限制最大最小值的
                        err_value.append(new_value)
                    # print(f"err_value = {err_value}")
                    fixed_params_dict[element] = err_value
        # print(f"self.lattice_template = {self.lattice_template}")
        # x_err = (-1 + 2 * random.random()) * self.max_error
        # y_err = (-1 + 2 * random.random()) * self.max_error
        print(f"After add error:SOL {fixed_params_dict}")

        self.reset_error(self.max_error)
        with open(self.lattice_error_template, "r",encoding="gb2312") as input, \
                open(self.lattice_template_out, "w",encoding="gb2312") as output:
            contents = input.read()
            for fixed_keyword in fixed_params_dict.keys():
                keyword_count = contents.count(fixed_keyword)
                print(f"keyword count = {keyword_count}")
                print(fixed_params_dict[fixed_keyword])
                # print(f"len(fixed_params_dict[fixed_keyword] = {len(fixed_params_dict[fixed_keyword])}")
                assert keyword_count == len(fixed_params_dict[fixed_keyword])
                for i in range(keyword_count):
                    contents = contents.replace(f"@{str(i + 1)}{fixed_keyword}$",
                                                str(fixed_params_dict[fixed_keyword][i]))
            output.write(contents)
        # print(contents)
        return self.lattice_template_out

    def reset_error(self, max_error):  # 随机添加原件偏差
        assert max_error == self.max_error
        print(f"===============================RESET ERROR {self.max_error}=============================")
        with open(self.lattice_template, "r",encoding="utf-8") as input:
            contents = input.read()

        x_err_list = []
        y_err_list = []

        for i in range(1, self.error_element_num + 1):
            if i < self.error_element_num:
                x_err = (-1 + 2 * random.random()) * max_error
                y_err = (-1 + 2 * random.random()) * max_error
            else:
                x_err = (-1 + 2 * random.random()) * max_error * self.MEBT_out_error_rate
                y_err = (-1 + 2 * random.random()) * max_error * self.MEBT_out_error_rate

            x_err_list.append(x_err)
            y_err_list.append(y_err)
            contents = contents.replace(f"@ERR_X{i}$", str(round(x_err, 3)))
            contents = contents.replace(f"@ERR_Y{i}$", str(round(y_err, 3)))
        print(f" x_err_list= {x_err_list}")
        print(f" y_err_list= {y_err_list}")
        with open(self.lattice_error_template, "w") as output:
            output.write(contents)

    @check_keys
    def generate_lattice_file(self, value_dict, lattice_file, lattice_template=None):

        if lattice_template is None:
            lattice_template = self.lattice_template_out

        with open(lattice_template, "r",encoding="gb2312") as input, \
                open(lattice_file, "w",encoding="gb2312") as out:
            contents = input.read()

            for key in value_dict.keys():
                keyword_count = contents.count(key)
                assert keyword_count == len(value_dict[key])
                for i in range(keyword_count):
                    contents = contents.replace(f"@{str(i + 1)}{key}$", str(round(value_dict[key][i], 4)))
            out.write(contents)

        return lattice_file


