from decisiontree.ID3Algorithm import ID3
id3_2 = ID3(dataset_train.csv,headers_train.csv,dataset_test.csv,headers_test.csv)
# dataset_train contains the training dataset with headers as headers_train
# dataset_test contains unlabled data with headers as headers_test
# all the agruments are of type list
id3_2.build_tree()
id3_2.classify()