def cleansing(text):
    
    import pandas as pd
    import json
    import re

    # Regrex
    text = text.lower() # membuat semua huruf menjadi lowercase
    text = text.strip() # menghapus spasi pada awal dan akhir kalimat
    text = re.sub('\n', ' ', text) # menghilangkan new line
    text = re.sub('x[a-z0-9]{,2}',' ', text) # menghilangkan rawstring emoji
    text = re.sub('user', ' ', text) # menghilangkan mention USER tertentu
    text = re.sub('url', ' ', text) # menghilangkan teks 'url' pada teks
    text = re.sub('http\S+', '', text) # menghilangkan url
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) # menghilangkan non-alpha numerik
    text = re.sub('( rt )|(rt )', ' ', text) # menghilangkan retweet
    text = re.sub('  +', ' ', text) # Menghilangkan spasi berlebih

    # Slangwords normalization
    with open('dataset/combined_slang_words.json') as f:
        jsonstr = json.load(f)
    df_slang = pd.DataFrame(list(jsonstr.items()),columns = ['slang','normal']) 

    Slang_dict = dict(zip(df_slang['slang'], df_slang['normal']))  # Membuat dataframe slang menjadi dict
    holder = [] # variabel list untuk menyimpan kata slang yang sudah dinormalkan
    
    #pengulangan untuk mengubah word slang menjadi word yang sesuai pada dict
    for word in text.split(' '): 
        
        if word in Slang_dict.keys(): 
            word = Slang_dict[word] # mengubah kata slang menjadi kata yang sesuai pada dict
            holder.append(word) # simpan kata slang yang sudah normal ke variabel holder
        else :
            holder.append(word) 
            
    text = ' '.join(holder) # mengembalikan satu kalimat yang sudah digabungkan dari list huruf pada holder


    # Stopwords removal
    df_stop = pd.read_csv('dataset/combined_stop_words.csv', encoding = 'latin-1', header = None)
    holder = []
      
    for words in text.split(' '):
        if words in df_stop[0].values:
          holder.append(' ')
        else:
          holder.append(words)

    text = ' '.join(holder)
    text = re.sub(' +', ' ', text) # menghilangkan spasi berlebih
    text = text.strip() # menghilangkan whitespace pada awal dan akhir string

    return text