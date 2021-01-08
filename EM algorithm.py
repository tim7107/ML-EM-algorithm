import numpy as np
import struct
##################################################################
######################Problem2####################################
##################################################################
"""
    Define function
"""
train_images_idx3_ubyte_file = 'C:/Users/tim/Desktop/碩一/碩一下/ML/HW02/train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = 'C:/Users/tim/Desktop/碩一/碩一下/ML/HW02/train-labels-idx1-ubyte'
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)
def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3通用function
    :param idx3_ubyte_file: idx3 file path
    :return: dataset
    """
    # 讀取2進位data
    bin_data = open(idx3_ubyte_file, 'rb').read()

    #解析head訊息
    offset = 0
    fmt_header = '>iiii'  #read 前4個 unsinged int 32 bit integer
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print ('魔数:%d, 圖片數量: %d张, 圖片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析data
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    #image:60000x(28*28)大小的ndarray
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '張')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((1,num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images
def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: dataset
    """
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print ('魔數:%d, 圖片數量: %d張' % (magic_number, num_images))

    # 解析data set
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '張')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def difference(mu, new_mu):
    ans = 0
    for i in range(10):
        for j in range(784):
            ans += abs(mu[i,j] - new_mu[i,j])
    
    return ans

def display_num(mean):
    for i in range(10):
        print("\nclass: ", i)
        for j in range(28):
            for k in range(28):
                if (mean[i][j*28 + k] > 0.5):
                    print("1", end=" ")
                else:
                    print("0", end=" ")
            print("")
        print("")
        
#########################################################
#####################Input and Setting###################
#########################################################
iteration = 0
N = 60000
flag = 1

print("Loading Data...")
train_images =  load_train_images()
train_labels = load_train_labels()
print(train_images.shape)
for i in range(60000):
    for j in range(784):
        if(train_images[i][j]>127):
            train_images[i][j]=1
        else:
            train_images[i][j]=0
print('Finish loading data!')
lamb = np.full(10, 0.1)
mu = np.random.rand(10, 784) #mean
weight = np.zeros(10)
print("Start calculating...")


while flag == 1 or iteration == 1000:
    
    w = np.zeros((N,10), dtype=float)
    most = 0
    label = []
    count_calculated = np.zeros(10, dtype=int)
    
#--------------------------------E-step--------------------------------------

    for i in range(N): #60000筆資料
        for j in range(10):
            weight[j] = lamb[j] #每次的weight set成lamb
            for k in range(784): # w = w *(mean)^(1 or 0) * (1 - mean)^(1 or 0)
                weight[j] *= (mu[j][k]**train_images[i][k]) * ((1 - mu[j][k])**(1 - train_images[i][k]))

        if sum(weight) > 0: #predict是60000筆資料的屬於哪類的機率
            w[i,:] = weight / sum(weight)
        else:
            w[i,:] = 0.1

        most = np.argmax(w[i,:]) #選擇最高機率者為當次label
        
        label.append(most) #將他加進label list裡
        count_calculated[most] += 1

#-------------------------------M-step----------------------------------------
    
    # update lambda
    z = [sum(w[:, i]) for i in range(10)] #update lambda
    lamb = z / sum(z)      
    new_mu = np.zeros((10,784)) #重新計算新的mean
    
    for i in range(10): #10類裡
        mean = np.zeros(784)
        for j in range(N):  #train_image的第一維(60000) 
            mean += w[j, i] * train_images[j]
        new_mu[i] = mean / z[i] 
        
    diff = float(difference(new_mu, mu))    
    display_num(new_mu)
    
#------------------------------條件判斷-----------------------------------------
    
    if diff < 10 or iteration == 1000:
        flag = 0
        break
    else:
        mu = new_mu
        iteration += 1
    print("----------------#", iteration, "iteration, differance: ", diff, "----------------")

#########################################################
#####################Showing Result######################
#########################################################
cm = np.zeros((10,10))
for i in range(N):
    cm[int(train_labels[i]),int(label[i])] += 1
    
for i in range(10):
    c = np.where(cm[i,:]==max(cm[i,:]))[0][0]
    if i < c or i == np.where(cm[:,c] == max(cm[:,c]))[0][0]:
        tmp = np.copy(cm[:,c])
        cm[:,c] = np.copy(cm[:,i])
        cm[:,i] = np.copy(tmp)
    
    
correct = 0
for i in range(10):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    row = 0
    col = 0
    TP = int(cm[i][i])
    for j in range(10):
        row += int(cm[i][j])
        col += int(cm[j][i])
    FN = row - TP
    FP = col - TP
    TN = N - ( TP + FN + FP)
    
    if (TP + FN) != 0:
        sens = TP / (TP + FN)
    else: 
        sens = 0
        
    if (FP + TN) != 0:
        spec = TN / (FP + TN)
    else:
        spec = 0
        
    print("\n-------------------------------------------------------\n")
    print("\nConfusion Matrix", i,  ":")
    print("\t\tPredict number ", i, "\tPredict not number", i)
    print("Is number ", i, end="\t\t")
    print(TP, "\t\t\t", FN)
    print("")
    
    print("Isn't number ", i, end="\t\t")
    print(FP, "\t\t\t", TN)
    print("\n\n")
    print("Sensitivity (Successfully predict number ",i , ") :", sens)
    print("Specificity (Successfully predict not number ",i , ") :", spec)
    print("\n\n")
    correct += TP

print('Total iteration to converge: ', iteration)
print('Total error rate: ', 1 - (correct / N))