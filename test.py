import load_dictionary as ld
import load_dataset as lds

en_dict, zh_dict = ld.load_dictionary()
print(en_dict.subwords[0:10])
print(zh_dict.subwords[0:10])
# print(next(iter(zh_dict)))

dataset = lds.load_dataset()
print(dataset)

