
# Beyond bags of features spatial pyramid matching for recognizing natural scene categories, CVPR 2006

논문 : https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf

데이터 로드
<pre><code>
labelll=pd.read_csv('../content/input/Label2Names.csv')
labelll.loc[100] = ['1', 'Faces']
labelll.loc[101] = ['102', 'BACKGROUND_Google']

print(labelll.iloc[0][1])
print(labelll)
</code></pre>
<pre><code>
#데이터 로드
data1 = ('../content/input/train')
train_data = list()
train_name = list()

namefile = os.listdir(data1)

#train
for file in namefile:
  imgfile = os.listdir(data1 + '/' + file)
  
  #(라벨링)
  for i in range (0,102):
    if (file == labelll.iloc[i][1]):
      label = labelll.iloc[i][0] #백그라운드 제외

  for img in imgfile:
    if (label == '102'):
      print("OK")
      break
    image = cv2.imread(data1 + '/' + file + '/' + img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256,256), interpolation = cv2.INTER_LINEAR)
    train_data.append(image)
    train_name.append(label)

train_data = np.array(train_data)
train_name = np.array(train_name)

</pre></code>

SIFT
<pre><code>
def siftfunction(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        step_size = 10 #(15->10)
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        #kp = sift.detect(img, None)
        des = sift.compute(img, kp)
        x.append(des[1])
        
    return x
</pre></code>

코드북
<pre><code>
all_train_desc = []
for i in range(len(x_train)):
    for j in range(x_train[i].shape[0]):
        all_train_desc.append(x_train[i][j,:])

all_train_desc = np.array(all_train_desc)

def build_codebook(X, voc_size):
   
    features = np.vstack((descriptor for descriptor in X))
    seeding = kmc2.kmc2(features, voc_size)
    kmeans = MiniBatchKMeans(n_clusters=voc_size, init=seeding).fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    return codebook
</pre></code>

SPM
<pre><code>
def build_spatial_pyramid(image, descriptor, level):
  
    step_size = 8
   
    height = int(image.shape[0] / step_size)
    width = int(image.shape[1] / step_size)

    idx_crop = np.array(range(len(descriptor))).reshape(height,width)
    size = idx_crop.itemsize

   
    bh, bw = 2**(5-level), 2**(5-level)
    shape = (int(height/bh), int(width/bw), bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])

    crops = np.lib.stride_tricks.as_strided(idx_crop, shape=shape, strides=strides)

    des_idxs = [col_block.flatten().tolist() for row_block in crops for col_block in row_block]

    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid

def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))

def input_vector_encoder(feature, codebook):
   
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist
    
x_train = [spatial_pyramid_matching(train_data[i],x_train[i], codebook, level=2) for i in range(len(train_data))]
</pre></code>


성능 향상을 위한 변경

-노이즈 데이터 제외
-kmeans 코드조각 개수 변경
-히스토그램 정규화
-파라미터 C 변경
-sift 스텝사이즈 변경
-SPM 적용
-SPM kenel 적용
-이미지 사이즈 변경


|SPM 적용| step_size | 파라미터 c | 코드조각 | kmc2 적용 | 성능 |
| --- | --- |--- |--- |--- |--- |
|x  | 15 | 0.0198 | 400 |   x |0.33333|
|x  | 15 | 0.0198 | 400 |   o |0.31914|
| x | 10 | 0.0198 | 400 |o  | 0.32151 |
| x | 10 | 0.0198 | 500 | o | 0.32860 |
| x | 10 | 0.0198 | 800 | o | 0.35106 |
| o | 15 | 0.0198 | 400 |   o |0.22163|
| o | 15 | 0.5 |400  |  o |0.22163|
| o | 10 | 0.0298 |400  |  o |0.22163|


-성능 향상의 큰 요인 : SPM kernel

|SPM 커널 적용|SPM 적용| step_size | 파라미터 c | 코드조각 | kmc2 적용 | 성능 |
| --- | --- |--- |--- |--- |--- |--- |
| x | o | 8 | 0.0298 | 400 | o |0.31796|
| o | o | 8| 0.0298 |400  |  o |0.59456|
| o | o | 8 | 0.0298 | 800 | o |0.61111|
