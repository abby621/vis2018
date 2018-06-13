import glob

train_ims = glob.glob('train/*/*.jpg')
train_ims_by_class = {}

for im in train_ims:
    cls = im.split('/')[1]
    if not cls in train_ims_by_class.keys():
        train_ims_by_class[cls] = {}
        train_ims_by_class[cls]['ims'] = []
        train_ims_by_class[cls]['count'] = 0
    train_ims_by_class[cls]['ims'].append(im)
    train_ims_by_class[cls]['count'] += 1

train_outfile = open('train.txt','a')
for cls in train_ims_by_class.keys():
    outstr = " ".join(train_ims_by_class[cls]['ims']) + '\n'
    train_outfile.writelines(outstr)

train_outfile.close()

test_ims = glob.glob('test/*/*.jpg')
test_ims_by_class = {}

for im in test_ims:
    cls = im.split('/')[1]
    if not cls in test_ims_by_class.keys():
        test_ims_by_class[cls] = {}
        test_ims_by_class[cls]['ims'] = []
        test_ims_by_class[cls]['count'] = 0
    test_ims_by_class[cls]['ims'].append(im)
    test_ims_by_class[cls]['count'] += 1

test_outfile = open('test.txt','a')
for cls in test_ims_by_class.keys():
    outstr = " ".join(test_ims_by_class[cls]['ims']) + '\n'
    test_outfile.writelines(outstr)

test_outfile.close()
