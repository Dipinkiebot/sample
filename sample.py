from src.emo1 import detect_emotion

x=[r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\angry\Training_3908.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\disgust\Training_2580532.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\fear\Training_301018.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\happy\Training_160942.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\neutral\Training_245243.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\sad\Training_577975.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\surprise\Training_353184.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\fear\Training_387737.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\angry\Training_872654.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\disgust\Training_19915172.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\happy\Training_1154381.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\neutral\Training_1586585.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\sad\Training_1357189.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\surprise\Training_2159810.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\angry\Training_2632144.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\disgust\Training_8819454.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\fear\Training_341092.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\happy\Training_160942.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\neutral\Training_242815.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\sad\Training_996099.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\angry\Training_364963.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\fear\Training_308765.jpg",r"C:\Users\ASUS\OneDrive\Desktop\project1\mockinter\mock\src\static\dataset\train\happy\Training_1548283.jpg"]
y=['0','1','2','3','4','5','6','2','0','1','3','4','5','6','0','1','2','3','4','5','0','2','3']
py=[]
for i in x:
    r=detect_emotion(i)
    py.append(r)
print(py)
print(y)



from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4','5','6']
print("For training")
print(classification_report(y, py, target_names=target_names))

from sklearn import metrics
cf=metrics.confusion_matrix(y,py)
print(cf)
dipin
