import os
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Função para carregar e pré-processar uma imagem
def carregar_imagem(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 300))
    return img

# Função para detectar a íris
def detectar_iris(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 1.5)
    edges = cv2.Canny(img_blur, 30, 100)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=20, minRadius=50, maxRadius=150)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            centro = (circle[0], circle[1])
            raio = circle[2]
            return centro, raio
    return None, None

# Função para extrair características usando LBP
def extrair_caracteristicas(img, centro, raio):
    mask = np.zeros_like(img)
    cv2.circle(mask, centro, raio, (255, 255, 255), thickness=-1)
    iris_segment = cv2.bitwise_and(img, img, mask=mask)
    lbp = feature.local_binary_pattern(iris_segment, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

# Função para reconhecimento de íris
def reconhecimento_iris(caminho_imagem, dataset, limiar=0.2):
    """
    Compara a imagem de teste com o dataset de íris.
    Se a menor distância for maior que o limiar, retorna uma mensagem de falha.
    """
    img = carregar_imagem(caminho_imagem)
    centro, raio = detectar_iris(img)
    if centro is None or raio is None:
        print("Íris não detectada.")
        return None

    features = extrair_caracteristicas(img, centro, raio)

    # Comparar com o dataset
    similaridades = []
    for label, data_features in dataset.items():
        distancia = euclidean(features, data_features)
        similaridades.append((label, distancia))

    # Ordenar por menor distância
    similaridades.sort(key=lambda x: x[1])

    # Verificar o menor valor de distância
    menor_distancia = similaridades[0][1]
    mais_semelhante = similaridades[0][0]

    # Normalizar a distância para calcular semelhanca
    # semelhanca = 1 - (distância normalizada pela maior possível)
    semelhanca = 1 - menor_distancia

    # Mostrar a imagem com a íris detectada
    plt.imshow(img, cmap='gray')
    if semelhanca >= (90 / 100):
        plt.title(f"Íris Detectada (Mais semelhante: {mais_semelhante})(semelhanca: {semelhanca:.2f})")
    else:
        plt.title("Íris não está presente na base de dados")
    if centro and raio:
        circle_img = cv2.circle(img.copy(), centro, raio, (255, 0, 0), 2)
        plt.imshow(circle_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Retornar resultado com base na semelhanca
    if semelhanca >= (90 / 100):
        print(f"Íris reconhecida: {mais_semelhante} (semelhanca: {semelhanca:.2f})")
        return mais_semelhante
    else:
        print("Essa íris não está presente na base de dados.")
        return None

# Criar um dataset de íris a partir de uma pasta
def criar_dataset_pasta(caminho_pasta):
    dataset = {}
    for arquivo in os.listdir(caminho_pasta):
        caminho_completo = os.path.join(caminho_pasta, arquivo)
        if os.path.isfile(caminho_completo) and arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = carregar_imagem(caminho_completo)
            centro, raio = detectar_iris(img)
            if centro and raio:
                features = extrair_caracteristicas(img, centro, raio)
                label = arquivo.split(".")[0]  # Nome do arquivo sem extensão como label
                dataset[label] = features
            else:
                print(f"Íris não detectada na imagem: {arquivo}")
    return dataset

# Função principal
def main():
    # Caminho da pasta contendo o dataset de íris
    caminho_pasta_dataset = 'archive/train/eye'  # Substitua pelo caminho da sua pasta
    dataset = criar_dataset_pasta(caminho_pasta_dataset)

    # Testar reconhecimento com uma nova imagem
    caminho_teste = 'img/exemploDatabase.JPG'
    reconhecimento_iris(caminho_teste, dataset)

if __name__ == "__main__":
    main()
