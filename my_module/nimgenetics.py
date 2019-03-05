#Import Libraries

import os
import fdb 
import shutil
import re
import pytesseract
from PIL import Image as IMG
from wand.image import Image
import PyPDF2
import nltk
from nltk.corpus import stopwords
from string import punctuation
import unicodedata
from CNIO_Tagger import CNIO_Tagger
from nltk import FreqDist

#Text Mining Functions

def load_files(root_dir):
    
    """
    root_dir = The main folder that start the folder-tree.
    This program returns two list:
    file_list = the absolute path to each pdf file
    l_pdf = all the pdf files 
    os.walk function is useful to get inside the tree and append needed files
    """
    
    file_list = []
    l_pdf = []
    
    for path, dirs, files in os.walk(root_dir):
        for d in dirs:
            input_folder = os.path.join(path, d)
            directory_file_list = os.listdir(input_folder)
            for filename in directory_file_list:
                suffix = 'pdf'
                if filename.endswith(suffix):
                    file_list.append(os.path.join(path, d, filename))
                    l_pdf.append(os.path.join(filename))
                    
    return file_list, l_pdf

def getIDs(id_list):
    
    r_id = []

    select = """

    select
        p.PETICIONCB
    
        from
            PETICION p
        
        where
            p.IDPETICION = '%s'
        
    """
    
    for number in id_list:
        query = select % number
        con = fdb.connect(dsn='192.168.30.207:G:\Gestlab\GestlabBD\GESTLAB.GLDB', user='MCARCAJONA', password='12345')
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        r_id.append(rows)
    
    return r_id

def identifiedPDF(folder_path, list_id, list_path):
    
    """
    list_id = a list of pdf identifiers that returns from gestlab
    list_path = the absolute path for each pdf filename where its name is not altered
    
    This program identified those pdf files that have a identifier from gestlab. Also introduces pdf files
    that are repeated with the same name and those which are discarted. These last are pdf files that did not fulfilled
    the conditions given by the pattern. e.g would be 8G or NT pdf files.
    """

    pdfI = folder_path+'Identificados/'
    pdfR = folder_path+'Repetidos/'
    pdfDes = folder_path+'Descartados/'
    
    id_pattern = '^1[3-9]N[R|S|G]'
    
    if not os.path.isdir(pdfI):
        os.mkdir(pdfI)
    
    if not os.path.isdir(pdfR):
        os.mkdir(pdfR)
    
    if not os.path.isdir(pdfDes):
        os.mkdir(pdfDes)
    
    for i in range(0, len(list_id)-1):
        
        if list_id[i]==list_id[i+1] and list_id[i] != 'sin_id':
            shutil.copy(list_path[i], pdfR+list_id[i]+'.pdf')
        
        elif re.search(pattern=id_pattern, string=list_id[i]):
            shutil.copy(list_path[i], pdfI+list_id[i]+'.pdf')
    
        else:
            shutil.copy(list_path[i], pdfDes+list_id[i]+'.pdf')

def nonIdentifiedPDF(folder_path, list_id, list_path, list_pdf):
    
    """
    This program is useful to extract those pdf files that dont return any identifier from gestlab.
    In this case the files are saved into a folder with the hexadecimal identificar without decoding.
    """
    
    pdfS = folder_path+'Sin_identificar/'
    
    if not os.path.isdir(pdfS):
        os.mkdir(pdfS)
    
    for i in range(0, len(list_id)-1):
        if list_id[i] == 'sin_id':
            shutil.copy(list_path[i], pdfS+list_pdf[i])  

def pdfToImage(f_path, pdf_list, fullPath):
    
    """
    This program is needed to convert the scanned pdf to a png image. The full path is the absolute path where
    the pdf files identified are located. Each image file will have the corresponding images depending on the 
    number of pages that the pdf contains.
    """
    
    image_folder = f_path+'Images/'
    
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    
    name = []
    for pdf in pdf_list:
        name.append(pdf.replace('.', '/').split('/')[-2])
        
    for i in range(0, len(fullPath)):
        
        try:
            opdf = Image(filename=fullPath[i], resolution=300)
            ipdf = opdf.convert('png')
        
            for idx, img in enumerate(ipdf.sequence):
                page = Image(image=img)
                page.save(filename=image_folder+name[i]+'_'+'image'+str(idx)+'.png')
                page.destroy()
                
            #print('Procesado ->', fullPath[i])
            ipdf.destroy()
            opdf.close()
            
        except:
            #print('No procesado ->', fullPath[i])
            pass
                
def ocrToImage(image):
    
    """
    This function transforms the image content to text that can be saved as txt format. 
    """
    text = pytesseract.image_to_string(IMG.open(image))
    return text

def makeCorpus(f_path, image_path):
    
    """
    This function takes a image block a transform it to a txt file. Each block contains as many images as the 
    the block has. Is because of that the extraction of the set identifiers and next searching for each name
    that is the same but, the tricky thing is that the program is executing the process every time depending on
    the range that is set up before the looping.
    
    Parameters:
    f_path = the original path where is located the main folder.
    image_path = the path where the images are located (image.png)
    """
    
    txt_folder = f_path+'textos/'
    if not os.path.isdir(txt_folder):
        os.mkdir(txt_folder)
        
    l_image = os.listdir(image_path)
    
    l_name = []
    for image in l_image:
        l_name.append(image.split('_')[0])
    
    l_set = set(l_name)
    l_unique = sorted(list(l_set))
    
    for i in range(0, len(l_unique)):
        
        try:
            images = []
            for image in l_image:
                id_image = image.split('_')
                if id_image[0] == l_unique[i]:
                    images.append(os.path.join(image_path, image))
        
            texts = []
            for img in images:
                text = ocrToImage(image=img)
                texts.append(text)
        
            s_text = ''.join(texts)
        
            outfile = str(l_unique[i])+'.txt'
            with open(txt_folder+outfile, 'w') as w:
                w.write(s_text)
        
            w.closed
            #print('Procesado:', l_unique[i])
        
        except:
            pass

def preprocessingText(f_path, t_path, min_len):
    
    """
    This function reduce the content of the text in orden to make easier the posterior analysis. 
    Should pay attention on text quality.
    Parameters:
    
    f_path = root-folder
    t_path = path where texts are stored
    """
    
    def elimina_tildes(cadena):
        s = ''.join((c for c in unicodedata.normalize('NFD', cadena) if unicodedata.category(c) != 'Mn'))
        return s
    
    # stopwords and punctuation for removal
    stop_words = set(stopwords.words('spanish'))
    punctuation_marks = set(punctuation)
    stop_words_punctuation_marks = stop_words.union(punctuation_marks)
    
    txt_folder = f_path+'c_texts/'
    if not os.path.isdir(txt_folder):
        os.mkdir(txt_folder)
    
    for i in range(0, len(t_path)):
        
        infile = t_path[i]
    
        with open(infile, 'r') as f:
        
            text = f.read()
    
            # split into tokens
            tokens = nltk.word_tokenize(text)
            tokens = [word.lower() for word in tokens]
    
            # remove stopwords and punctuation marks
            # remove all tokens that are not alphabetic
            words = [word for word in tokens if word not in stop_words_punctuation_marks]
            words = [word for word in words if word.isalpha()]
    
            # Extract words with minimun or maximum length
            min_words = [word for word in words if len(word)>=min_len]
            fdist = FreqDist(min_words)
            hapaxes = fdist.hapaxes()
    
            # eliminar tildes
            texto = ' '.join(hapaxes)
            texto = elimina_tildes(texto)
        
        f.closed
        
        outfile = str(infile.split('/')[-1])
        with open(txt_folder+outfile, 'w') as w:
            w.write(texto)
        w.closed

def getANN(f_path, f_texts):
    
    """
    Obtain the ann files for each txt. We use the CNIO tagger.
    Take into account the filtering used by filterANN.
    """
    #shutil.copytree(src=scr_path, dst=dst_path)
    ann_folder = f_path+'ann_folder/'
    if not os.path.isdir(ann_folder):
        os.mkdir(ann_folder)
    
    t_path = f_texts
    textos = os.listdir(t_path)
    l_textos = []
    
    for file in textos:
        l_textos.append(os.path.join(t_path, file))
    
    for i in range(0, len(l_textos)):
        
        infile = l_textos[i]
        with open(infile, 'r') as f:
            s = ''
            l = []
            lines = f.readlines()
            for line in lines:
                s_line = line.replace('\n', ' ')
                l.append(s_line)
            s+=''.join(l)
        
        f.closed
        
        name = infile.replace('.', '/')
        name = name.split('/')[-2]
        
        tag = CNIO_Tagger()
        result = tag.parse(text=s)
        tag.write_brat(parsed=result, folder_path=ann_folder+name+'.ann')

def filterANN(scr_path, dst_path, ann_path, filterALL="YES"):
    
    """
    This function copy the text folder and add the ann files but filtered or not by NC and AQ.
    Must be provided:
    scr_path = path to text folder.
    dst_path = path to the new folder that contains also texts and .ann files.
    ann_path = path where are ann files non-filtered.
    filterALL = apply or not a defined filter
    """
    
    shutil.copytree(src=scr_path, dst=dst_path)
    
    ann_path = ann_path
    l_ann = os.listdir(ann_path)
    annPath = []
    for file in l_ann:
        annPath.append(os.path.join(ann_path, file))
    
    if filterALL is "YES":
        
        for i in range(0, len(l_ann)):
            infile = annPath[i]
            with open(infile, 'r') as f:
                s = ''
                f_ann = []
                lines = f.readlines()
                for line in lines:
                    s_line = line.split()
                    if s_line[1] == 'NC' or s_line[1] == 'AQ':
                        f_ann.append(line)
                s+=''.join(f_ann)
            f.closed
            outfile = infile.split('/')[-1]
            with open(dst_path+outfile, 'w') as w:
                w.write(s)
            w.closed
    
    if filterALL is "NO":
        
        for i in range(0, len(l_ann)):
            infile = annPath[i]
            with open(infile, 'r') as f:
                s = ''
                f_ann = []
                lines = f.readlines()
                for line in lines:
                    f_ann.append(line)
                s+=''.join(f_ann)
            f.closed
            outfile = infile.split('/')[-1]
            with open(dst_path+outfile, 'w') as w:
                w.write(s)
            w.closed