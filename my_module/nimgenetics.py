# Importar librerias #

import os
import fdb 
import shutil
import re
import codecs
import itertools
import pytesseract
from PIL import Image as IMG
from wand.image import Image
import nltk
from nltk.corpus import stopwords
from string import punctuation
import unicodedata
from CNIO_Tagger import CNIO_Tagger
from nltk import FreqDist

# Función para obtener el absolute path #

def absolutePath(dpath):
    path = dpath
    l_path = os.listdir(path)
    l_abs = []
    for ele in l_path:
        l_abs.append(os.path.join(path, ele))
    
    return l_abs

# ------------------------------------------ 3. SISTEMA, DISEÑO Y DESARROLLO --------------------------------------------------- #

#### Funciones de preprocesamiento de los informes clínicos ####

## 3.2 Obtención de informes clínicos de la base de datos de NIMGenetics ##

# 3.2.1 Mapeo de los informes localizados en la carpeta principal #

def load_files(root_dir):
    
    """
    La función load_files utiliza el la función walk del módulo os para mapear el árbol de carpetas.
    root_dir = Carpeta principal donde se encuentran los informes distribuidos como un árbol de directorios.
    El programa devuelve dos listas:
    file_list = El path absoluto para cada informe (PDF).
    l_pdf = Todos los PDF. Únicamente el nombre que los identifica.
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

# 3.2.2 Obtención de los identificadores reales de los informes #

def getIDs(id_list):
    
    """
    La función getIDs permite obtener los identificadores reales de los informes.
    Utiliza una query que apunta a una tabla de la base de datos de la cual se obtienen los ID reales.
    La query requiere de una dirección dsn, nombre de usuario y password. No indicados por seguridad.
    El parámetro id_list es la lista de PDF mapeados.
    El programa devuelve una lista de identificadores correctos para cada informe/paciente.
    """
    
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
        con = fdb.connect(dsn='XXX', user='XXX', password='XXX')
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        r_id.append(rows)
    
    return r_id

# 3.2.3 Almacenamiento de los informes con su identificador correcto #

def identifiedPDF(folder_path, list_id, list_path):
    
    """
    La función identifica los PDF cuyo identificador se encuentre en la base de datos y son almacenados
    en una carpeta denominada "Identificados".También se almacenan los informes que estén repetidos y aquellos 
    que se descarten por no cumplir el patrón. Estos son almacenados en las carpeta "Repetidos" y "Descartados".
    En los casos de "Identificados" y "Repetidos" el identificador obtenido de la base de datos es el nombre
    utilizado para reemplazar el identificador en hexadecimal inicial. En el caso de los "Descartados" no es posible
    obtener el identificador de la base de datos y se almacena el PDF con el identificador hexadecimal.
    
    Parámetros:
    
    folder_path = carpeta origen donde están los PDF almacenados.
    list_id = lista de identificadores reales obtenidos de la base de datos.
    list_path = path absoluto donde se localizan los informes.
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

def emptyPDF(folder_path, list_id, list_path, list_pdf):
    
    """
    Esta función extrae los documentos que no tienen identificador en la base de datos.
    En este caso los informes serán guardados con su identificador hexadecimal original.
    Los parámetros utilizados son identicos al programa anterior a excepción de list_pdf.
    list_pdf = lista de identificadores en hexadecimal.
    """
    
    pdfS = folder_path+'sin_identificar/'
    
    if not os.path.isdir(pdfS):
        os.mkdir(pdfS)
    
    for i in range(0, len(list_id)):
        if list_id[i] == 'sin_id':
            shutil.copy(list_path[i], pdfS+list_pdf[i])  

## 3.3 Conversión de los informes a texto plano para la extracción de terminología médica ##

# 3.3.1 Generación de imágenes en formato PNG a partir de PDF escaneados #
            
def pdfToImage(f_path, pdf_list):
    
    """
    Esta función es útil para convertir PDF escaneados en imánes en formato PNG.
    Se generan tantas imagenes como páginas tenga el PDF a convertir.
    
    Parámetros:
    
    f_path = path de la carpeta donde guardar las imágenes.
    pdf_list = lista de PDF escaneados.
    """
    
    image_folder = f_path+'Images/'
    
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    
    name = []
    for pdf in pdf_list:
        name.append(pdf.replace('.', '/').split('/')[-2])
        
    for i in range(0, len(pdf_list)):
        
        try:
            opdf = Image(filename=pdf_list[i], resolution=300)
            ipdf = opdf.convert('png')
        
            for idx, img in enumerate(ipdf.sequence):
                page = Image(image=img)
                page.save(filename=image_folder+name[i]+'_'+'image'+str(idx)+'.png')
                page.destroy()
                
            #print('Procesado ->', pdf_list[i])
            ipdf.destroy()
            opdf.close()
            
        except:
            #print('No procesado ->', pdf_list[i])
            pass
                

# 3.3.2 Generación de texto plano a partir de imágenes en formato PNG #

def ocrToImage(image):
    
    """
    Esta función permite transformar el contenido de la imagen a texto plano mediante la aplicación de ténicas OCR de Python.
    El programa toma como parámetro una imagen la cual es convertida a texto.
    """
    
    text = pytesseract.image_to_string(IMG.open(image))
    return text

def makeCorpus(f_path, image_path):
    
    """
    Esta función toma las imágenes correspondientes a un informe (PDF). El número de páginas del PDF indica el número de imágenes
    que serán transformadas a texto. El proceso se ejecuta utilizando un set único de identificadores los cuales se corresponden
    con el nombre de las imágenes. Este nombre siempre es el mismo teniendo en cuenta el PDF de procedencia.
    El bucle vuelve a comenzar cuando las imágenes de un informe se han procesado y prosigue con el siguiente identificador.
    
    Parámetros:
    
    f_path = path donde se guardarán los textos generados.
    image_path = path donde se localizan las imágenes.
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

        
# ------------------------------------- 4. EXPERIMENTOS REALIZADOS Y RESULTADOS ------------------------------------------------ #

#### Funciones de extracción terminológica ####

## 4.2 Extracción de terminología mediante Cutext ##

def getCutextResults(cutext_in, textos_ori, old, new):
    
    """
    Esta función adapta la funcionalidad de Cutext para procesar documentos de texto plano.
    Utiliza el archivo jar que viene con la herramienta para su correspondiente ejecución.
    
    Parámetros:
    
    cutext_in = localización del directorio in dentro de Cutext.
    textos_ori = path absoluto de los textos a procesar.
    old = path del output generado por la ejecución de Cutext.
    new = nueva localización donde se almacena el output.
    """
 
    os.chdir(cutext_in)
    for i in range(0, len(textos_ori)):
        shutil.copy(textos_ori[i], cutext_in)
        command = 'java -jar path_to_/cutext.jar -TM -generateTextFile true -inputFile path_to_/cutext/in/{}' \
                  .format(textos_ori[i].split('/')[-1])
        os.system(command=command)
        shutil.move(old, new+textos_ori[i].split('/')[-1])


# 4.2.1 Elaboración de un corpus final a partir de los resultados de cutext #  

def cutextCorpus(lista_cutext, path_to_words):
    
    """
    Esta función extrae los términos "Term" del output generado por Cutext.
    Se genera un corpus de términos almacenado en path_to_words.
    
    Parámetros:
    
    lista_cutext = todos los documentos de texto generados por la ejecución de cutext.
    path_to_words = path donde guardar el corpus final después de procesar el output de cutext.
    """
    corpus_cutext = []
    
    for i in range(0, len(lista_cutext)):
        
        infile = lista_cutext[i] 
        
        with open(infile, 'r') as f:
            lines = f.readlines()
            concepts = []
            for line in lines:
                if line.startswith('Term'):
                    concepts.append(line.replace('\n', '').split(': ')[1])
        f.closed
        
        corpus_cutext.append(concepts)
        flat = list(itertools.chain.from_iterable(corpus_cutext))
        #print(str(lista_cutext[i]))
        
    outfile = 'corpus_cutext.txt'
    with open(path_to_words+outfile, 'w') as w:
        w.write('\n'.join(flat))
    w.closed

# 4.3 Extracción de terminología mediante la estrategia lookup #

def lookup(textos_ori, megadict, path_to_words):
    
    """
    Esta función permite generar un corpus mediante la técnica lookup.
    Esta estrategia está definida para ejecutarse a nivel de frase.
    
    Parámetros:
    
    textos_ori: path absoluto donde se localizan los textos originales.
    megadict: path donde se localiza el megadiccionario.
    path_to_words: path donde se guardará el corpus generado.
    """
    
    # Parte 1: documentos que cumplen el patrón de exonim -> infiles
    infiles = []
    for i in range(0, len(textos_ori)):
        infile = textos_ori[i]
        with open(infile, 'r') as f:
            lines = f.readlines()
            #print('infile open:', infile)
            for line in lines:
                if line.startswith('PRUEBA SOLICITADA: EXONIM'):
                    infiles.append(infile)
                    #print('infile matched:', infile)
        f.closed
    
    # Parte 2: cargar el megadiccionario -> items_cleaned
    with codecs.open(megadict, 'r', 'utf-8') as f:
        items = f.readlines()
    items_cleaned = []
    for item in items:
        items_cleaned.append(item.replace('\n','').lower())
    del(items)
    f.closed
    
    # Parte 3: extraer las frases de cada infile -> flat 
    corpus_lookup = []
    for infile in infiles:
        with open(infile, 'r') as f:
            lines = f.readlines()
            text = []
            for line in lines:
                text.append(line.replace('\n','').lower())
            text = list(set(text))
        f.closed
        #print('infile procesado:', infile)
    
        corpus_lookup.append(text)
    
    flat = list(itertools.chain.from_iterable(corpus_lookup))
    
    # Parte 4: ejecutar el lookup -> mapped_flat
    mapped_sentences = []
    for sentence in flat:
        found = []
        for item in items_cleaned:
            # check if concept is inside the sentence
            if item in sentence:
                found.append(item)  
        #print('item extraido:', str(sentence))
                
        mapped_sentences.append(found)

    mapped_flat = list(itertools.chain.from_iterable(mapped_sentences))
    
    # Parte 5: guardar el corpus generado por lookup
    outfile = 'corpus_lookup_procesado.txt'
    with open(path_to_words+outfile, 'w') as w:
        w.write('\n'.join(mapped_flat))
    w.closed

# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

# FUNCIONES NO INCLUIDAS EN EL TRABAJO #
# DESARROLLO DE PRUEBAS PARA PERSPECTIVAS FUTURAS #

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
