
עבודה 30 אחוז נדב מנירב

העבודה שעשיתי יכולה להבדיל ולהפריד בין תמונות של אריה לבין תמונות של אריה ים, באמצעות אלגוריתמים של קייאנאן ופרספטרון.
בחרתי את הנושא הזה מכיוון שאני מאוד אוהב חיות, ורציתי לראות את ההבדל המובהק בין הכחולים של הים בתמונות של אריה הים לבין הכתומים של הספארי בתמונות של האריה. בנוסף אחת מהתמונות של האריות זו תמונה שצילמתי בעצמי כשהייתי בטנזניה

אימפורטים מפייתון
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

"""# Start of KNN

הפעולה מקבלת תמונה ומחזירה ממוצע צבעים שלה.
היא מחשבת את המימדים של התמונה ועוברת על כל הפיקסלים שלה ומוסיפה לקאונטרים את כמות האדום, הירוק והכחול שיש בכל פיקסל. לבסוף היא מחלקת את כמות הצבעים הללו שהתקבלו מכל הפיקסלים ומחלקת במספר הפיקסלים כדי לקבל את ממוצע האדום, הירוק והכחול בתמונה
"""

#מקבלת תמונות ומחזירה ממוצע של הצבעים: אדום ירוק וכחול

def image_to_features(image):

  red_counter = 0 #כמה אדום
  green_counter = 0#כמה ירוק
  blue_counter = 0 #כמה כחול
  counter = 0 #כמות הפיקסלים בתמונה

  (height,width,color) = image.shape #המימדים של התמונה

  #עובר פיקסל-פיקסל בתמונה
  for h in range(height):
    for w in range(width):
      counter=counter+1
      #הוספת כמות הצבעים בתמונה
      red_counter += image[h,w,0]
      green_counter += image[h,w,1]
      blue_counter += image[h,w,2]

  average_red = red_counter/counter #ממוצע האדום בתמונה
  average_green = green_counter/counter #ממוצע הירוק בתמונה
  average_blue = blue_counter/counter #ממוצע הכחול בתמונה

  features = [average_red,average_green,average_blue] #מערך עם הצבעים של התמונה
  return features

"""מימוש הפעולה שכתבנו להשגת ממוצע הצבעים של כל התמונות: אריה, אריה ים וטסט"""

featuresT=np.zeros((38,3)) #יצירת מטריצה של 38 שורות (כמספר התמונות) ושלושה טורים כמספר הצבעים

colors=[] #הצבעים של הנקודות על הגרף
labelT=[] #השם של התמונה: אריה, אריה ים או טסט

for i in range(38):
  if i<15:# אריה
    image=mpimg.imread("/content/drive/MyDrive/DATA/Project/lion"+str(i+1)+".jpg") #lion 1-15
    featuresT[i]=image_to_features(image)
    labelT.append("lion")
    colors.append("green")

  elif i>=15 and i<30 :# אריה ים
    image=mpimg.imread("/content/drive/MyDrive/DATA/Project/sea"+str(i-14)+".jpg") #seaLion 1-15
    featuresT[i]=image_to_features(image)
    labelT.append("seaLion")
    colors.append("red")
  else:
    image=mpimg.imread("/content/drive/MyDrive/DATA/Project/test"+str(i-29)+".jpg")
    featuresT[i]=image_to_features(image)
    labelT.append("test")
    colors.append("blue")

# הצגת הנתונים
print(featuresT)
print(colors)
print(labelT)

"""יצירת גרף מהצבעים אדום וירוק.
נקודה אדומה - אריה ים.
נקודה ירוקה - אריה.
נקודה כחולה - טסט
"""

plt.scatter(x=featuresT[:,0],y=featuresT[:,1],color=colors) #הכנסת הנקודות לגרף
plt.title("red and green") #כותרת
plt.xlabel("red") #כותרת ציר איקס
plt.ylabel("green") #כותרת ציר וואי
plt.show() #הצגת הגרף

"""יצירת הגרף מהצבעים אדום וכחול, אותו נבחר להכנת להכנת האלגוריתמים מכיוון שבו ההפרדה הטובה ביותר בין הנקודות הירוקות והאדומות"""

plt.scatter(x=featuresT[:,0],y=featuresT[:,2],color=colors) #הכנסת הנקודות לגרף
plt.title("red and blue") #כותרת
plt.xlabel("red") #כותרת ציר איקס
plt.ylabel("blue") #כותרת ציר וואי
plt.show() #הצגת הגרף

"""יצירת גרף מהצבעים ירוק וכחול"""

plt.scatter(x=featuresT[:,1],y=featuresT[:,2],color=colors) #הכנסת הנקודות לגרף
plt.title("green and blue") #כותרת
plt.xlabel("green") #כותרת ציר איקס
plt.ylabel("blue") #כותרת ציר וואי
plt.show() #הצגת הגרף

"""פעולה שמחשבת את המרחקים בין שתי נקודות, כדי למצוא בשלב יותר מאוחר את ה"שכנים" של כל טסט

הגרף שבחרנו קודם, רלוונטי רק לאלגוריתמים האחרים, עבור הקייאנאן הפעולה הזו פועלת באופן תלת מימדי ועוברת על כל הצבעים של התמונה.
"""

def distance(p1,p2):
  d=0
  for i in range(len(p1)):
    d+=(p1[i]-p2[i])**2 #לפי נוסחת מרחק בין שתי נקודות במרחב
    d=np.sqrt(d)
  return d

"""יצירת שלושה מערכים אחד בתוך השני. המערך הגדול "אולדיסטנסס" מכיל בתוכו 8 מערכי "דיסטנסס" כל תא בשביל כל טסט. בתוך כל תא במערך הזה יש את המערך "דיס אנד לאב" שבו יש את המרחק של כל טסט מכל 30 הנקודות, והשם של הנקודה ממנה מודדים את המרחק, אריה או אריה ים."""

allDistances=[] #מערך כולל
for x in range(8): #עבור כל טסט
  distances = []
  for i in range(30): #עבור כל אחת מהתמונות של אריה ואריה ים
    dis=distance(featuresT[x+30],featuresT[i])
    label=labelT[i]
    dis_and_lab=[dis,label]
    distances.append(dis_and_lab)
  allDistances.append(distances)
print(allDistances)

"""מסדרים את המרחקים במערכים "דיסטנסס" מהקטן לגדול. כדי למצוא את המרחקים הכי קטנים בשביל לחזור את סוג הטסט. אריה או אריה ים."""

for i in range(8):
  allDistances[i].sort(key=lambda x:x[0])
print(allDistances)

"""חיזוי האלגוריתם באמצעות לקיחת שלושת הנקודות בעלי המרחקים הקטנים ביותר מהטסט והסתכלות על הלייבל שלהם."""

neighbors=[]
predictionsListKNN3=[]
predictionsListKNN5=[]
predictionsListKNN7=[]
k=3 #מתחילים מ3 שכנים, עושים גם 5 ו7
for y in range(3):
  print(k,"neighbors")
  for x in range(8): #עבור כל טסט
    for i in range(k):
      neighbors.append(allDistances[x][i][1]) #הוספת הלייבל של השכנים הקרובים
    prediction=max(neighbors,key=neighbors.count) #חיזוי הטסט, לפי הלייבל של רוב השכנים הקרובים.
    print(prediction)

    if k==3:
      predictionsListKNN3.append(prediction)
    elif k==5:
      predictionsListKNN5.append(prediction)
    else:
      predictionsListKNN7.append(prediction)

    neighbors=[]
  k=k+2   #  הגדלת הפרמטר בשביל הרצת האלגוריתם עבור 5 ו7 שכנים
print(predictionsListKNN3)
print(predictionsListKNN5)
print(predictionsListKNN7)

"""Confusion Matrix for k=3"""

actual = ['lion', 'lion', 'lion', 'lion', 'seaLion', 'seaLion', 'seaLion', 'seaLion']
predicted = predictionsListKNN3

print(predicted)
print(actual)
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = ConfusionMatrixDisplay(confusion_matrix, display_labels=['lion', 'seaLion'])
cm_display.plot(cmap = 'Blues')
print(classification_report(actual, predicted, target_names=['lion', 'seaLion']))

"""Confusion matrix k=5"""

# actual = ['lion', 'lion', 'lion', 'lion', 'seaLion', 'seaLion', 'seaLion', 'seaLion']
predicted = predictionsListKNN5

print(predicted)
print(actual)
confusion_matrix_KNN5 = metrics.confusion_matrix(actual, predicted)
cm_display_KNN5 = ConfusionMatrixDisplay(confusion_matrix_KNN5, display_labels=['lion', 'seaLion'])
cm_display_KNN5.plot(cmap = 'Blues')
print(classification_report(actual, predicted, target_names=['lion', 'seaLion']))

"""confusion matrix k=7"""

# actual = ['lion', 'lion', 'lion', 'lion', 'seaLion', 'seaLion', 'seaLion', 'seaLion']
predicted = predictionsListKNN7

print(predicted)
print(actual)
confusion_matrix_KNN5 = metrics.confusion_matrix(actual, predicted)
cm_display_KNN5 = ConfusionMatrixDisplay(confusion_matrix_KNN5, display_labels=['lion', 'seaLion'])
cm_display_KNN5.plot(cmap = 'Blues')
print(classification_report(actual, predicted, target_names=['lion', 'seaLion']))

"""# start of Perceptron

הפעולה מקבלת תמונות ומחזירה ממוצע רק של הצבעים אדום וכחול, כי אלה הצבעים שמייצגים את הגרף שבחרנו.
"""

#מקבלת תמונות ומחזירה ממוצע של הצבעים: אדום וכחול

def image_to_features_red_blue(image):

  red_counter = 0 #כמה אדום
  blue_counter = 0 #כמה כחול
  counter = 0 #כמות הפיקסלים בתמונה

  (height,width,color) = image.shape #המימדים של התמונה

  #עובר פיקסל-פיקסל בתמונה
  for h in range(height):
    for w in range(width):
      counter=counter+1
      #הוספת כמות הצבעים בתמונה
      red_counter += image[h,w,0]
      blue_counter += image[h,w,2]

  average_red = red_counter/counter #ממוצע האדום בתמונה
  average_blue = blue_counter/counter #ממוצע הכחול בתמונה

  features = [average_red,average_blue] #הכנסת הצבעים לרשימה
  return features

"""הפרדה בין התמונות הרגילות לבין הטסטים לשתי מטריצות שונות, שבשתיהן יהיו רק הצבעים אדום וכחול"""

featuresTNoTestsRedBlue=np.zeros((30,2)) #יצירת מטריצה של 30 שורות ושתי עמודות שבכל תא יש מספר 0
testsRedBlue=np.zeros((8,2)) #מטריצה של 8 שורות ושתי עמודות שבכל תא יש 0
colorsNoTests=[] #הצבעים של הנקודות של התמונות ללא הטסטים

for i in range(30):
  colorsNoTests.append(colors[i])

for i in range(38):
  if i<15:# אריה
    image=mpimg.imread("/content/drive/MyDrive/DATA/Project/lion"+str(i+1)+".jpg") #lion 1-15
    featuresTNoTestsRedBlue[i]=image_to_features_red_blue(image)

  elif i>=15 and i<30 :# אריה ים
    image=mpimg.imread("/content/drive/MyDrive/DATA/Project/sea"+str(i-14)+".jpg") #seaLion 1-15
    featuresTNoTestsRedBlue[i]=image_to_features_red_blue(image)
  else: #tests
    image=mpimg.imread("/content/drive/MyDrive/DATA/Project/test"+str(i-29)+".jpg")
    testsRedBlue[i-30]=image_to_features_red_blue(image)
print(featuresTNoTestsRedBlue)
print(testsRedBlue)

"""פעולה לחישוב אם נקודה מסוימת היא מתחת או מעל לקו. בשימוש בפעולה שבונה את הקו כדי לתקן אותו בעזרת מקדם הלמידה, ובשימוש לאחר שהקו נבנה לחיזוי הטסטים."""

def perceptron_predict(w,bias,x):
  output=0
  for i in range(len(w)):
    output=output+w[i]*x[i]
  output=output+bias
  if output>0:
    return 1 #מתחת לקו
  else:
    return -1 #מעל הקו

"""פעולה לחישוב המקדמים האופטימליים של הפרספטרון"""

#פעולה שמקבלת רשימת נקודות והסיווג של כל נקודה לאחד או מינוס אחד
#הפעולה מחזירה את המקדמים האופטימליים של הפרספטרון
def perceptron_fit(featuresT,labels):
  n_examples,n_features=featuresT.shape
  w=np.ones((n_features)) #ניחוש ראשוני
  print(w)
  bias=1 #ניחוש ראשוני
  learning_rate=0.1 #קבוע הלמידה
  num_epocs=50

  #run until no errors
  for epoch in range(num_epocs):
    n_errors=0

    #run over all samples
    for i in range(n_examples):
      x=featuresT[i]
      y=labels[i]

      prediction=perceptron_predict(w,bias,x) #הפעלת הפרספטרון

      #if false negative (error on positive example)
      if y==1 and prediction==-1:
        n_errors+=1
        for i in range(len(w)):
          w[i]=w[i]+learning_rate*x[i]
        bias=bias-1

      #if false positive (error in negative examples)
      if y==-1 and prediction==1:
        n_errors+=1
        for i in range(len(w)):
          w[i]=w[i]-learning_rate*x[i]
        bias=bias-1

    print('epoch=',epoch,'n_errors=',n_errors)
    if n_errors==0:
      break
  return[w,bias] #החזרת המקדמים האופטימליים

labelTNoTestsNum=labelT[0:30] #הלייבלים ללא הטסטים
for i in range(len(labelTNoTestsNum)): # שינוי אריה ל1 ואריה ים למינוס 1 כי הפעולה של חישוב המקדמים של הפרספטרון עובדת עבור 1 ומינוס 1
  if labelTNoTestsNum[i]=="lion":
    labelTNoTestsNum[i]=1
  else:
    labelTNoTestsNum[i]=-1
print(perceptron_fit(featuresTNoTestsRedBlue,labelTNoTestsNum))

"""ציור הקו של הפרספטרון"""

w, bias = perceptron_fit(featuresTNoTestsRedBlue , labelTNoTestsNum) #קבלת המקדמים האופטימליים
#ציור הקו
x0 = featuresTNoTestsRedBlue[:,0].min()
x1 = featuresTNoTestsRedBlue[:,0].max()
y0 = -(x0*w[0]+bias)/w[1]
y1 = -(x1*w[0]+bias)/w[1]
plt.scatter(x=featuresT[:,0],y=featuresT[:,2],color=colors)
plt.plot([x0,x1], [y0,y1])
plt.show() #הצגת הקו

"""Prediction based on the perceptron. 1=Lion, -1=Sea Lion"""

predictionsListPerceptron=[]
for i in range(8):
  print(perceptron_predict(w,bias,testsRedBlue[i]))
  predictionsListPerceptron.append(perceptron_predict(w,bias,testsRedBlue[i]))

"""confusion matrix perceptron"""

# actual = ['lion', 'lion', 'lion', 'lion', 'seaLion', 'seaLion', 'seaLion', 'seaLion']
actual=[1,1,1,1,-1,-1,-1,-1]
predicted = predictionsListPerceptron

print(predicted)
print(actual)
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = ConfusionMatrixDisplay(confusion_matrix, display_labels=['lion', 'seaLion'])
cm_display.plot(cmap = 'Blues')
print(classification_report(actual, predicted, target_names=['lion', 'seaLion']))

"""![image.png](data:image/png;base64,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)

# start of SVM
"""

for i in range(len(labelTNoTestsNum)): #האלגוריתם מקבל 1 ו 0 ועד עכשיו היה לנו 1 ומינוס 1
  if(i>=15):
    labelTNoTestsNum[i]=0

"""The use of SVC

ציור הקו המפריד והחיזוי של הטסטים. 1 בשביל אריה, 0 בשביל אריה ים
"""

all_predicts = []
for num in [1.0,0.1,0.01]:
  svm=SVC(kernel='linear', C=num) #יצירת המסווג
  svm.fit(featuresTNoTestsRedBlue,labelTNoTestsNum) #אימון המסווג, תווית 1 בשביל אריה, תווית 0 בשביל אריה ים
  predict=svm.predict(testsRedBlue) #חיזוי המסווג
  distance=svm.decision_function(testsRedBlue) #הפעולה מחזירה מרחק מהקו, חיובי אם היא מעל הקו ושלילי אם מתחתיו


  svm=SVC(kernel='linear', C=num) #יצירת המסווג
  svm.fit(featuresTNoTestsRedBlue,labelTNoTestsNum) #אימון המסווג, תווית 1 בשביל אריה, תווית 0 בשביל אריה ים
  predict=svm.predict(testsRedBlue) #חיזוי המסווג
  distance=svm.decision_function(testsRedBlue) #הפעולה מחזירה מרחק מהקו, חיובי אם היא מעל הקו ושלילי אם מתחתיו
  plt.scatter(featuresTNoTestsRedBlue[:,0], featuresTNoTestsRedBlue[:,1], color=colorsNoTests) #הוספת הנקודות
  plt.scatter(testsRedBlue[:,0],testsRedBlue[:,1],color='blue')
  #get the minimal and maximal X and Y values in the current plot
  ax=plt.gca()
  xlim=ax.get_xlim()
  ylim=ax.get_ylim()
  xx=np.linspace(xlim[0],xlim[1],30)
  yy=np.linspace(ylim[0],ylim[1],30)

  #create grid distance
  XX=np.zeros((30,30))
  YY=np.zeros((30,30))
  Dist=np.zeros((30,30))
  for row in range(30):
    for col in range(30):
      XX[row,col]=xx[col]
      YY[row,col]=yy[row]
      Dist[row,col]=svm.decision_function([(XX[row,col],YY[row,col])])

  plt.scatter(XX,YY,Dist,marker='*')
  plt.scatter(XX,YY,-Dist,marker='*')

  #plot line on change from positive to negative in Dist image
  ax.contour(XX,YY,Dist,colors='k',
            levels=[-1,0,1], alpha=0.5, #draw 3 lines, center and two offsets by +-1
            linestyles=['--','-','--']) #style of the 3 lines
  plt.show()
  print(predict)
  all_predicts.append(predict)

"""confusion matrix svm"""

# actual = ['lion', 'lion', 'lion', 'lion', 'seaLion', 'seaLion', 'seaLion', 'seaLion']
for predict in all_predicts:
  actual=[1,1,1,1,0,0,0,0]
  predicted = predict

  print(predicted)
  print(actual)
  confusion_matrix = metrics.confusion_matrix(actual, predicted)
  cm_display = ConfusionMatrixDisplay(confusion_matrix, display_labels=['lion', 'seaLion'])
  cm_display.plot(cmap = 'Blues')
  print(classification_report(actual, predicted, target_names=['lion', 'seaLion']))

"""רפלקציה: אני נהניתי מאוד לעשות את העבודה. לפני שהתחלתי אותה הרגשתי שאני לא בקיא בחומר ב100 אחוז, אבל לאחר שביצעתי אותה, נעזרתי באינטרנט, דיברתי עם חברים וחשבנו ביחד על בעיות שהיו לנו בקוד, אני מרגיש עכשיו שאני מבין את החומר הרבה הרבה יותר. לפני העבודה קצת נרתעתי מפייתון ובמיוחד מזה שיש המון פקודות שאני לא מכיר, ועכשיו למדתי להשתמש ולהכיר המון פקודות וקוד חדש שלא הכרתי, ואני הרבה יותר נהנה מלעבוד בשפה

העבודה הייתה חוויה מאוד מאתגרת. בתחילה, למדתי את התאוריות והמושגים הבסיסיים של כל אלגוריתם. ניתן היה למצוא מקורות מידע מגוונים ומדריכים עשירים שהסבירו את הרעיונות והמתמטיקה של כל אחד מהאלגוריתמים הללו.

סיכום: בעבודה הפרדתי בין תמונות של אריה לבין תמונות של אריה ים, תמונות אלו היוו את הדאטה.
השתמשתי בתחילה באלגוריתם קייאנאן שאיתו הצלחתי למצוא את השכנים הקרובים לכל אחת מהתמונות של הטסטים. ובעקבות זה אני עשיתי חיזוי לגבי סוג התמונה.
לאחר מכן המשכתי למסווג של פרספטרון. איתו בניתי קו שיפריד בין קבוצת הנקודות של האריות לבין אלה של אריות הים לפי חיזוי ראשוני. מה שמעל הקו היה אריה ים, ומתחת אליו - אריה. לבסוף השתמשתי במסווג של אסויאם שאיתו בניתי קו בצורה יותר מדוייקת עם שוליים.
"""