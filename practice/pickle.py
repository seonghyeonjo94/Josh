import pickle

# 파일 저장하기
with open('data.pickle', 'wb') as f:
    pickle.dump(fs, f)

# 파일 불러오기
with open('data.pickle', 'rb') as f:
    fs = pickle.load(f)