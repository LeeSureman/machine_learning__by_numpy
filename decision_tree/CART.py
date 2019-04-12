import copy
import math
import treePlotter

node_id = 0
def createTestSet(indexize=False):
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    if indexize:
        testSet = [[0, 1, 0, 0],
                   [0, 2, 1, 0],
                   [2, 1, 1, 0],
                   [0, 1, 1, 1],
                   [1, 1, 0, 1],
                   [1, 0, 1, 0],
                   [2, 1, 0, 1]]
    else:
        testSet = [['sunny','mild','high','false'],
                   ['sunny','cool','normal','false'],
                   ['rain','mild','normal','false'],
                   ['sunny','mild','normal','true'],
                   ['overcast','mild','high','true'],
                   ['overcast','hot','normal','false'],
                   ['true','rain','mild','high']]

    return testSet


def createDataSet(indexize=False):
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    if indexize:
        dataSet = [[0, 0, 0, 0, 'N'],
                   [0, 0, 0, 1, 'N'],
                   [1, 0, 0, 0, 'Y'],
                   [2, 1, 0, 0, 'Y'],
                   [2, 2, 1, 0, 'Y'],
                   [2, 2, 1, 1, 'N'],
                   [1, 2, 1, 1, 'Y']]
    else:
        dataSet = [['sunny', 'hot', 'high', 'false', 'N'],
                     ['sunny', 'hot', 'high', 'true', 'N'],
                     ['overcast', 'hot', 'high', 'false', 'Y'],
                     ['rain', 'mid', 'high', 'false', 'Y'],
                     ['rain', 'cool', 'normal', 'true', 'Y'],
                     ['rain', 'cool', 'normal', 'false', 'N'],
                     ['overcast', 'cool', 'normal', 'true', 'Y']]
        # feat_list = ['outlook', 'temperature', 'humidity', 'windy']

    labels = ['outlook', 'temperature', 'humidity', 'windy']
    # labels = set(labels)
    return dataSet, labels


class TreeNode:
    def __init__(self):
        self.attribute = None
        self.value = None
        self.go = {}
        self.label = None
        self.default = None
        self.id = 0

def print_node(t_node):
    print(t_node.id,t_node.attribute,t_node.value,[p.id for p in t_node.go.values()],t_node.label,t_node.default)

    for p in t_node.go.values():
        print_node(p)

def divide_by_label(data):
    label2examples = dict()

    for i, example in enumerate(data):
        if label2examples.get(example[-1]) is None:
            label2examples[example[-1]] = [example]
        else:
            label2examples[example[-1]].append(example)

    return label2examples

def cal_information_entropy(examples):
    label2examples = divide_by_label(examples)
    divided = label2examples.values()
    ie = 0
    sum = 0
    for examples in divided:
        sum += len(examples)
    for examples in divided:
        p = len(examples) / sum
        ie -= p * math.log(p,2)

    return ie

def divide_by_a_name(data,a_index):
    value2examples = dict()

    for i, example in enumerate(data):
        if value2examples.get(example[a_index]) is None:
            value2examples[example[a_index]] = [example]
        else:
            value2examples[example[a_index]].append(example)

    return value2examples

def divide_by_a_name_and_value(data,a_index,a_value):
    value2examples = dict()

    for i, example in enumerate(data):
        if value2examples.get(example[a_index]==a_value) is None:
            value2examples[example[a_index]==a_value] = [example]
        else:
            value2examples[example[a_index]==a_value].append(example)

    return value2examples

def cal_information_entropy_gain(divided):
    example_num_sum = 0
    result = 0

    for examples in divided.values():
        example_num_sum+=len(examples)

    for examples in divided.values():
        result-=(len(examples)/example_num_sum)*cal_information_entropy(examples)

    return result


def cal_information_entropy_gain_ratio(divided):
    example_num_sum = 0
    result = 0

    iv = 0
    for examples in divided.values():
        example_num_sum += len(examples)

    for examples in divided.values():
        result -= (len(examples) / example_num_sum) * cal_information_entropy(examples)
        iv -= (len(examples) / example_num_sum) * (math.log(len(examples) / example_num_sum,2)+1e-5)

    return result / iv

def cal_gini(examples):
    divided = divide_by_label(examples)
    example_num_sum = 0
    result = 1
    for examples in divided.values():
        example_num_sum += len(examples)

    for examples in divided.values():
        result-=(1-(len(examples)/example_num_sum)**2)

    return result

def cal_gini_index(divided):
    example_num_sum = 0
    result = 0
    for examples in divided.values():
        example_num_sum+=len(examples)

    for examples in divided.values():
        result+=cal_gini(examples)*len(examples)/example_num_sum

    return result

def select_best_attribute_id3(data, a_names, a2index):
    best_ie_gain = -1e20
    best_a_name = None
    for a_name in a_names:
        tmp_divided = divide_by_a_name(data, a2index[a_name])
        tmp_ie_gain = cal_information_entropy_gain(tmp_divided)
        if tmp_ie_gain > best_ie_gain:
            best_ie_gain = tmp_ie_gain
            best_a_name = a_name

    return best_a_name

def select_best_attribute_c45(data, a_names, a2index):
    best_ie_gain_ratio = -1e20
    best_a_name = None
    for a_name in a_names:
        tmp_divided = divide_by_a_name(data, a2index[a_name])
        tmp_ie_gain_ratio = cal_information_entropy_gain_ratio(tmp_divided)
        if tmp_ie_gain_ratio > best_ie_gain_ratio:
            best_ie_gain = tmp_ie_gain_ratio
            best_a_name = a_name

    return best_a_name

def select_best_attribute_cart(data, a_names, a2index):
    best_gini_index = 1e20
    best_a_name = None
    best_a_value = None
    a2values = dict()
    index2a = dict()

    for a,index in a2index.items():
        index2a[index] = a

    for a_name in a_names:
        a2values[a_name] = set()
    for i,example in enumerate(data):
        for j,value in enumerate(example[:-1]):
            # if index2a.get(j) is not None:
            if index2a[j] in a_names:
                a2values[index2a[j]].add(value)




    for a_name in a_names:
        for value in a2values[a_name]:
            tmp_divided = divide_by_a_name_and_value(data, a2index[a_name],value)
            tmp_gini_index = cal_gini_index(tmp_divided)
            if tmp_gini_index < best_gini_index:
                best_gini_index = tmp_gini_index
                best_a_name = a_name
                best_a_value = value

    #cart是二叉树，所以返回最优划分属性，和属性最优划分值
    return best_a_name,best_a_value


def grow(t_node, data, a_names, a2index,select_algorithm,isBinary=False):
    #grow函数，使t_node对应data训练，划分完之后就递归

    global node_id
    node_id+=1
    t_node.id = node_id
    a_names = copy.deepcopy(a_names)
    label_set = set()
    for i, example in enumerate(data):
        label_set.add(example[-1])
        if len(label_set) > 1:
            break
    #first case
    if len(label_set) == 1:
        t_node.label = data[0][-1]
        return

    # second case
    tag = True
    for a in a_names:
        a_index = a2index[a]
        a_set = set()
        for i, example in enumerate(data):
            a_set.add(example[a_index])

        if len(a_set) > 1:
            tag = False
            break
        if not tag:
            break

    label2freq = dict()
    for i, example in enumerate(data):
        label2freq[example[-1]] = label2freq.setdefault(example[-1], 0) + 1
    major_label = 0
    max_freq = 0
    for label, freq in label2freq.items():
        if freq > max_freq:
            max_freq = freq
            major_label = label

    if tag:
        t_node.label = major_label
        return

    # third case
    if not isBinary:
        best_a = select_algorithm(data, a_names, a2index)
        a_names.remove(best_a)
        t_node.attribute = best_a
        a_index = a2index[best_a]

        t_node.default = major_label

        value2examples = dict()

        for i, example in enumerate(data):
            if value2examples.get(example[a_index]) is None:
                value2examples[example[a_index]] = [example]
            else:
                value2examples[example[a_index]].append(example)

        for value, examples in value2examples.items():
            t_node.go[value] = TreeNode()
            grow(t_node.go[value], examples, a_names, a2index,select_algorithm)
    else:
        best_a,best_a_value = select_algorithm(data, a_names, a2index)
        t_node.value = best_a_value
        t_node.attribute = best_a
        a_index = a2index[best_a]
        t_node.default = major_label

        value2examples = dict()
        values = set()
        for i, example in enumerate(data):
            # print(example[a_index])
            values.add(example[a_index])
            if value2examples.get(example[a_index]==best_a_value) is None:
                value2examples[example[a_index]==best_a_value] = [example]
            else:
                value2examples[example[a_index]==best_a_value].append(example)


        t_node.go[True] = TreeNode()
        #是这个最优划分属性的最优值的分支
        #最优属性的最优值分支，必定只有最优属性的最优值，所以不要再考虑该属性了
        a_names.remove(best_a)
        grow(t_node.go[True], value2examples[True], a_names, a2index,select_algorithm,isBinary)
        a_names.add(best_a)
        #若这个属性的值种类小于等于二，则说明，在最优属性的非最优值分支，该属性的值小于一种，所以不需要考虑该属性
        if len(values)<=2:
            a_names.remove(best_a)

        if value2examples.get(False)!=None:
            #是最优划分属性的非最优值的分支
            t_node.go[False] = TreeNode()
            grow(t_node.go[False], value2examples[False], a_names, a2index, select_algorithm, isBinary)

class DecisionTree:
    def __init__(self, a2index,type_='cart'):
        if type_=='cart':
            self.binary = True
        else:
            self.binary = False
        self.root = TreeNode()
        self.a2index = a2index
        self.a_names = set(a2index.keys())
        self.select_algorithm_dict = {'cart':select_best_attribute_cart,'c4.5':select_best_attribute_c45,
                                 'id3':select_best_attribute_id3}

        self.select_algorithm = self.select_algorithm_dict[type_.lower()]
    def train_self(self,data):
        grow(self.root,data,self.a_names,self.a2index,self.select_algorithm,isBinary=self.binary)

    def predict(self,example):
        p = self.root
        if not self.binary:
            while(True):
                if p.label is not None:
                    return p.label

                a_index = self.a2index[p.attribute]
                if p.go.get(example[a_index]) is None:#对应于形成只带预测结果的叶子结点
                    return p.default
                else:
                    p = p.go[example[a_index]]
        else:
            while(True):
                if p.label is not None:
                    return p.label

                a_index = self.a2index[p.attribute]

                if p.go.get(example[a_index]==p.value) is None:
                    return p.default
                else:
                    p = p.go[example[a_index]==p.value]








def transform_my_tree_to_fang(t_node):
    result = {}
    a_name = t_node.attribute
    result[a_name] = {}
    if t_node.label!=None:
        return t_node.label
    result[a_name][''+str(t_node.value)] = transform_my_tree_to_fang(t_node.go[True])
    if False in t_node.go:
        result[a_name]['not '+str(t_node.value)] = transform_my_tree_to_fang(t_node.go[False])
    else:
        result[a_name]['not '+str(t_node.value)] = t_node.default

    return result

if __name__ == '__main__':

    # 数据格式，同方老师的文件
    # 每一个example是[属性1，属性2，属性3……label]

    data,attributes = createDataSet()
    test_data = createTestSet()
    a2index = dict()
    for i,a in enumerate(attributes):
        a2index[a] = i

    dt = DecisionTree(a2index,type_='cart')
    dt.train_self(data)

    train_x = []
    for e in data:
        train_x.append(e[:-1])

    print("train:")
    for example in train_x:
        print(dt.predict(example))

    print("test:")
    for example in test_data:
        print(dt.predict(example))


    print_node(dt.root)
    fang_tree = transform_my_tree_to_fang(dt.root)
    treePlotter.createPlot(fang_tree)