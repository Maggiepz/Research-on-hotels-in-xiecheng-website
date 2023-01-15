#encoding=utf-8
#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib
import matplotlib.pyplot as plt
#指定字体
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

factors_list = ['价格','房间面积','床数','免费专车','交通便利','绿化环境好']
#导入数据
path = "dataset\\"
data = pd.read_excel(path + "携程酒店.xlsx")

#KMeans聚类
# 导入KMeans库
kmeans_data = data[factors_list]
from sklearn.cluster import KMeans
# KMeans模型初始化构建，设置init='k-means++',algorithm='full'
kmeans_model = KMeans(init='k-means++', n_clusters=3, algorithm='full')
# KMeans模型训练
kmeans_model.fit(kmeans_data)
# 获得聚类中心
cluster_center = kmeans_model.cluster_centers_
print(cluster_center)
# KMeans模型预测
pred = kmeans_model.predict(kmeans_data)
pred = pd.DataFrame(pred)

# pred_zero = len(pred[pred==0])
# pred_one = len(pred[pred==1])
data['class'] = pred
# if pred_zero > pred_one:
#     data['class'] = data['class'].replace([0,1],[1,0])
# # data.to_excel("data.xlsx",index=None)

#划分训练集和测试集
from sklearn.model_selection import train_test_split
X = data.loc[:,factors_list]
y = data.loc[:,'评价']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = X,X,y,y
#数据标准化
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def FeatureSelect(RF,data, label, png_save_path,total_ratio=0.8):
    import matplotlib.pyplot as plt
    # 采用随机森林进行特征选择
    FeatureNames = data.columns
    data = data.values
    feature_num = np.array(data).shape[1]
    label = label.values
    RF.fit(data, label)
    importances = RF.feature_importances_
    indices = np.argsort(importances)[::-1]
    Select_Num = feature_num
    for i in range(feature_num):
        if sum(importances[indices[:i]]) > total_ratio:
            Select_Num = i + 1
            break
    indices = indices[:Select_Num]
    data_selected = pd.DataFrame(data[:, indices])
    importances_selected = pd.DataFrame(importances[indices])
    FeatureNames_selected = FeatureNames[indices].tolist()
    data_selected = pd.DataFrame(data_selected)
    data_selected.columns = FeatureNames_selected
    #打印
    # for f in range(data_selected.shape[1]):
    #     # 给予10000颗决策树平均不纯度衰减的计算来评估特征重要性
    #     print("%s %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
    df_res = pd.DataFrame()
    df_res['featureName'] = FeatureNames_selected
    df_res['importance'] = importances_selected
    #画图
    plt.title('Feature Importance')
    plt.bar(range(data_selected.shape[1]), importances[indices], color='orange', align='center')
    plt.xticks(range(data_selected.shape[1]), FeatureNames_selected, rotation=90,fontproperties = myfont)
    plt.xlim([-1, data_selected.shape[1]])
    plt.tight_layout()
    plt.savefig(png_save_path)
    plt.show()
    return data_selected, importances_selected, FeatureNames_selected,df_res

best_params= {'n_estimators': 100,
              'max_depth': 9, 'min_samples_leaf':2}

#特征选择
feature_name = data.columns
feat_labels = data.columns
data = pd.DataFrame(X_train,columns=factors_list)
label = y_train
png_save_path = "png/Importance_01.png"
print("####################################决策树模型####################################")
# from sklearn.ensemble.forest import RandomForestClassifier
# RF = RandomForestClassifier(n_estimators=150,
#                                 max_features="sqrt",
#                                 min_samples_leaf=4,
#                                 n_jobs=4, random_state=0)
from sklearn import tree
RF = tree.DecisionTreeClassifier(criterion="gini",
                 splitter="best",
                 max_depth=5,
                 min_samples_split=2)
RF.fit(data, label)
data_selected, importances_selected, FeatureNames_selected,df_res = FeatureSelect(RF,data, label,png_save_path,total_ratio=1)
# print(FeatureNames_selected)
df_res.to_excel("data_out/Importance_List_01.xls",index=None)
from sklearn.metrics import confusion_matrix
# 模型评估
y_pred = RF.predict(X_test)
test_df = pd.DataFrame(X_test,columns=factors_list)
test_df['class'] = y_test
test_df['class_pred'] = y_pred

print('准确率:')
print(RF.score(X_test, y_test))
print('混淆矩阵:')
print(confusion_matrix(y_test,y_pred))

#绘图
dot_data =tree.export_graphviz(
        RF,
        out_file = None,
        feature_names = factors_list,
        filled = True,
        impurity = False,
        rounded = True
    )
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
graph.write_png("out.png")  #当前文件夹生成out.png

# from sklearn.manifold import TSNE
# digits_proj = TSNE(random_state=0).fit_transform(kmeans_data)    #将X降到2维
# import matplotlib
# RS = 20190101 #Random state
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted') #调色板颜色温和
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
# import matplotlib.patheffects as PathEffects
#
# def scatter(x, colors):
#     palette = np.array(sns.color_palette("hls", 5))
#     f = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
#                     c=palette[colors.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
#     plt.xlabel("x")
#     plt.ylabel('y')
#     plt.title('KMeans')
#     ax.axis('off')
#     ax.axis('tight')
#     # 给类群点加文字说明
#     txts = []
#     for i in range(5):
#         xtext, ytext = np.median(x[colors == i, :], axis=0)  # 中心点
#         txt = ax.text(xtext, ytext, str(i), fontsize=24)
#         txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])  # 线条效果
#         txts.append(txt)
#     return f, ax, sc, txts#
#
# scatter(digits_proj, y_test)
# plt.savefig('digits_tsne-generated.png', dpi=120)
# plt.show()






