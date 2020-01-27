from train_and_eval_utils import CNNDailyMailDataset
from pytorch_transformers import XLNetTokenizer

best_list = [("1a0cb94420e0e1ce99d2e67607175aa067d2adae", "../data/dev_data0-9/dev_data_5"),
             ("76f3f120cf62c5fc7a0cd0c95d3245775e41834a", "../data/dev_data0-9/dev_data_6"),
             ("3bb5c44400b711cb49c9e0ad025ae674a81ea43f", "../data/dev_data0-9/dev_data_6")]

worst_list = [("eaeaab1f32b0bd58ccbadc269331da95a227b77a", "../data/dev_data0-9/dev_data_3"),
              ("121fd3cf9ce903f515797dc1a9e7ffe03daf5820", "../data/dev_data0-9/dev_data_7"),
              ("0d593c3807c979dfd4b0a0a380ce911e9e07e2f4", "../data/dev_data0-9/dev_data_7")]

best_list2 = [("3bb5c44400b711cb49c9e0ad025ae674a81ea43f","../data/dev_data0-9/dev_data_6"),
              ("692781308ebf134e5e2aaac2ced718d0bba6bada","../data/dev_data0-9/dev_data_7"),
              ("64c48d1888ed32f5adc887275a87a488f5a5fcc8", "../data/dev_data0-9/dev_data_1")]

worst_list2 = [("093b975f7dffb6040ffee0729233f473de0f82db", "../data/dev_data0-9/dev_data_4"),
               ("4358634a59d0aad941da3d9997650d40665f6d79", "../data/dev_data0-9/dev_data_8"),
               ("8fbee4db6fc56a137a2b87991337d8fa066817dc", "../data/dev_data0-9/dev_data_5")]

for article, origin in worst_list2:
    t = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    dat = CNNDailyMailDataset(tokenizer=t, data_dir=origin)
    storyname = next(st for st in dat.stories_path if article in st)
    index = dat.stories_path.index(storyname)
    _, story, summary = dat[index]
    story_encoded = t.encode(" ".join(story))[:67]
    print(t.decode(story_encoded))
    print(" ".join(summary))
