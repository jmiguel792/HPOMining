# Importar librerias #

import os
import fdb 
import shutil
import re
import codecs
import itertools
import collections
import pandas as pd
import pytesseract
from PIL import Image as IMG
from wand.image import Image
import nltk
from nltk.corpus import stopwords
from string import punctuation
import unicodedata
from CNIO_Tagger import CNIO_Tagger
from nltk.probability import FreqDist

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

# 4.5 Elaboración de un corpus global a partir de cutext y lookup

def mergeCorpus(p_cutext, p_lookup, path_to_words):
    
    """
    Esta función genera un corpus final a partir de dos listas de vocabulario.
    Utiliza como parámetros los path donde estén localizados los set de conceptos.
    """
    
    with open(p_cutext, 'r') as f:
        lines_cutext = f.readlines()
        #print(len(lines_cutext))
    f.closed
    
    with open(p_lookup, 'r') as f:
        lines_lookup = f.readlines()
        #print(len(lines_lookup))
    f.closed
    
    # Extraer el vocabulario común junto con el no común
    lines_global = sorted(list(set(lines_cutext) | set(lines_lookup)))
    
    outfile = 'corpus_global.txt'
    
    with open(path_to_words+outfile, 'w') as w:
        w.write(''.join(lines_global))
    w.closed

# 4.5.1 Representación de clínica sobre una matriz de distribución terminológica

def mapeoTerminos(l_textos, infile_global):
    
    """
    Esta función nos permite generar una matriz de mapeo para representar la distribución de la clínica
    de cada paciente en función de la terminología integrada en el corpus global.
    
    Parámetros:
    
    l_textos: path de los textos originales de los informes.
    infile_global: path donde se encuentra el corpus global.
    """
    
    # parte 1: trabajar con los exonim -> infiles
    infiles = []
    for i in range(0, len(l_textos)):
        infile = l_textos[i]
        with open(infile, 'r') as f:
            lines = f.readlines()
            #print('infile open:', infile)
            for line in lines:
                if line.startswith('PRUEBA SOLICITADA: EXONIM'):
                    infiles.append(infile)
                    #print('infile matched:', infile)
        f.closed
    
    # parte 2: utilizar el vocabulario del corpus global -> vocabulario
    with open(infile_global, 'r') as f:
        vocabulario = []
        lines = f.readlines()
        for ele in lines:
            vocabulario.append(ele.replace('\n',''))
        del(lines)
    f.closed
    
    # parte 3: construir el dataFrame -> mapeado_terminos.xlxs
    final_list = []
    for i in range(0, len(infiles)):
        
        # añadir -> vocabulario
        # open the infile -> items
        with open(infiles[i], 'r') as f:
            lines = f.readlines()
            items = []
            for line in lines:
                items.append(line.replace('\n','').lower())
            del(lines)
        f.closed
    
        # mapping words
        mapped = []
        for line in items:
            found = []
            for ele in vocabulario:
                if ele in line:
                    found.append(ele)
            
            mapped.append(found)
    
        # compact the mapped list
        mapped_flat = list(itertools.chain.from_iterable(mapped))
        
        # prepare data for DataFrame
        counter = collections.Counter(mapped_flat)
        infile_mapeado = [infiles[i].replace('.','/').split('/')[-2]]
        items_counter = list(counter.items())
        infile_mapeado.extend(items_counter)
        final_list.append(infile_mapeado)
    
    # make a DataFrame from list of lists to excel document
    df = pd.DataFrame(data=final_list)
    df.to_excel(excel_writer='mapeado_terminos.xlsx', header=False, index=False)

# 4.5.2 Selección de 500 términos para el mapeo manual a HPO

def subsetManual(c_uncleaned, c_global):
    
    """
    Esta función genera dos diccionarios en orden ascendente e inverso respecto del conteo de términos 
    a nivel global utilizando el corpus_lookup sin curar y ejecutando la técnicalookup entre este corpus
    y el global curado para así obtener este análisis. Esta estrategia permite también obtener una selección
    de los 500 términos con menos frecuencia para utilizarlos para el mapeo manual a HPO.Los parámetros son 
    los path hacia el corpus_lookup sin curar y el corpus global generado con la función mergeCorpus.
    """
    
    # parte 1: texto sin curar -> texto
    with open(c_uncleaned, 'r') as f:
        texto = []
        lines = f.readlines()
        for line in lines:
            texto.append(line)
        del(lines)
    f.closed 
    
    # parte 2: corpus global -> texto_p
    with open(c_global, 'r') as f:
        texto_p = []
        lines = f.readlines()
        for line in lines:
            texto_p.append(line)
        del(lines)
    f.closed
    
    # parte 3: lookup para el conteo -> mapped
    mapped = []
    for line in texto:
        found = []
        for item in texto_p:
            if item in line:
                found.append(item)
                #print('item found:', item)
        mapped.append(found)
    
    # compactar mapped -> mapped_flat
    mapped_flat = list(itertools.chain.from_iterable(mapped))
    
    # eliminar saltos de página -> mapped_final
    mapped_final = []
    for ele in mapped_flat:
        mapped_final.append(ele.replace('\n',''))
    
    # parte 4: conteo de términos
    counter = collections.Counter(mapped_final)
    
    # diccionario ordenado por valor
    d = counter.items()
    
    # diccionario en orden ascendente al conteo de términos
    d_ord_increasing = collections.OrderedDict(sorted(sorted(d), key=lambda t:t[1]))
    
    # diccionario en orden descendente al conteo de términos
    d_ord_decreasing = collections.OrderedDict(sorted(sorted(d), key=lambda t:t[1], reverse=True))
    
    # DataFrame to write excel by ascending or descending order
    df = pd.DataFrame(data=d_ord_decreasing, index=['count']).T
    df1 = pd.DataFrame(data=d_ord_increasing, index=['count']).T
    df.to_excel(excel_writer='conteo_terminos_increasing.xlsx')
    df1.to_excel(excel_writer='conteo_terminos_decreasing.xlsx')
    
    # parte 5: extracción 500 términos para validación manual
    fdist = FreqDist(mapped_final)
    all_items = fdist.items()
    terms = fdist.most_common()[-500:]
    
    terminos = []
    for i in terms:
        terminos.append(i[0])
        
    # parte 6: guardar los términos seleccionados para el mapeo manual
    outfile = 'términos_mapeo_manual.txt'
    with open(outfile, 'w') as w:
        w.write('\n'.join(terminos))
    
    return all_items


# 4.6 Exomiser

def makeTemplate(patient_id, lista_hpo, infile, exomiser_path):
    
    """
    Esta función adapta los yml necesarios para ejecutar exomiser.
    
    Parámetros:
    
    patient_id: el id de los pacientes para completar la ruta input.
    lista_hpo: HPOs asociados a la terminología extraída de los informes.
    infile: yml base adaptable.
    exomiser_path: path donde almacenar los resultados de la ejecución de exomiser.
    """
    
    for i in range(0, len(patient_id)):
        
        #input
        vcf_name = 'vcf:'
        vcf_info = '/home/exomiser-9/vcf/{}_Haplo_raw.snps.indels.vcf'.format(patient_id[i])
        vcf_path = [vcf_name, vcf_info]
    
        #HPOs
        hpoIds = 'hpoIds:'
        hpo_info = hpoIds + ' ' + str(lista_hpo[i])
    
        #output
        output = 'outputPrefix:'
        output_info = '/home/exomiser-9/vcf/{}_output'.format(patient_id[i])
        output_path = [output, output_info]
    
        # open the template infile
        with open(infile, 'r') as f:
            s_text = f.read()
            sr_text = s_text.replace(vcf_name, ' '.join(vcf_path)) \
                            .replace(hpoIds, hpo_info) \
                            .replace(output, ' '.join(output_path))
        f.closed
       
        # write outfile
        outfile = patient_id[i] + '.yml'
        with open(exomiser_path+outfile, 'w') as w:
            w.write(sr_text)
        w.closed

# 4.6.2 Resultados de la ejecución de exomiser y comparativa con Health29

def filterVariants(data):
    
    """
    Esta función está diseñada para elaborar un dataframe a partir de los resultados de exomiser.
    Filtra algunos genes complejos y devuelve las 10 primeros genes candidatos calculados por
    exomiser probables para explicar el fenotipo del paciente.
    
    Parámetro:
    
    data = path donde se almacena el output de exomiser.
    """
    
    # df principal
    df = pd.read_csv(data, delimiter='\t', usecols=[0,1,8,10,28,29,30,31])
    df = df.rename(columns={'#CHROM':'chr','POS':'pos','FUNCTIONAL_CLASS':'functional_class',
                            'EXOMISER_GENE':'gene','EXOMISER_VARIANT_SCORE':'variant_score',
                            'EXOMISER_GENE_PHENO_SCORE':'pheno_score',
                            'EXOMISER_GENE_VARIANT_SCORE':'gene_variant_score',
                            'EXOMISER_GENE_COMBINED_SCORE':'gene_combined_score'})
  
    df_filter = df[(df.variant_score != 0)&
                   (df.gene != 'HLA-DRB1')&
                   (df.gene != 'HYDIN')&
                   (df.gene != 'HERC2')]
    
    return df_filter.head(10)

def locVariant(data, gen_sel):
    
    """
    Función que localiza el gen candidato indicado por los analistas en relación con la lista
    de variantes calculada por exomiser.
    
    Parámetros:
    
    data = path donde se almacena el output de exomiser.
    gen_sel = gen candidato principal.
    """
    
    # df principal
    df = pd.read_csv(data, delimiter='\t', usecols=[0,1,8,10,28,29,30,31])
    df = df.rename(columns={'#CHROM':'chr','POS':'pos','FUNCTIONAL_CLASS':'functional_class',
                            'EXOMISER_GENE':'gene','EXOMISER_VARIANT_SCORE':'variant_score',
                            'EXOMISER_GENE_PHENO_SCORE':'pheno_score',
                            'EXOMISER_GENE_VARIANT_SCORE':'gene_variant_score',
                            'EXOMISER_GENE_COMBINED_SCORE':'gene_combined_score'})
    
    # df de localización
    gen_pos = gen_sel
    df_loc = df.loc[df.gene == gen_pos]
    
    return df_loc

# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #

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
