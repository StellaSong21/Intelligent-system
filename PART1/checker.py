alpha = 0.01
gold_file = open('gold.txt', 'r').readlines()
pred_file = open('./record/result/res_' + str(alpha) + '.txt', 'r').readlines()
count = 0
if len(pred_file) < 1800:
    print('预测数据少于1800')
else:
    for pred, gold in zip(pred_file[:1800], gold_file):
        if pred.strip() == gold.strip():
            count += 1
    print(alpha, ", 准确率 : {}".format(count / 1800))
