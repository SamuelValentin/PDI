# Disciplina de Processamentos digital de imagens:

Desenvolvimento dos trabalhos para fins acadêmicos:

## 1 - Segmentação: 
Nota 100/100

Para realizar esse trabalho foi utilizada a linguagem python e a biblioteca openCV.
O objetivo deste trabalho era contar a quantidade de grãos de arroz em uma imagem disponibilizada pelo professor.

O primeiro passo desse trabalho foi binarizar a imagem em duas classes, o que seria o fundo e o que seria o arroz. Depois usamos a função FloodFill de maneira recursiva para catalogar os componentes conexos que possivelmente seriam grãos de arroz enquanto percorremos a imagem. Por fim, tentamos tirar os ruídos da imagem que foram catalogados como arroz, verificando se o tamanho do componente conexo respeita o tamanho mínimo que foi estipulado.  

Para rodar o arquivo "main.py" que esta no diretorio "Trabalho1-Segmentacao", basta ter a imagem "arroz.bmp" dentro da mesma pasta e usar o comando:

"$ python main.py"

## 2 - Filtro média: 
Para realizar esse trabalho foi utilizada a linguagem python e a biblioteca openCV.

O objetivo deste trabalho era desenvolver 3 algoritmos para realizar o filtro da média:
a) Filtro Ingênuo
  Essa abordagem era a mais simples e menos eficiente. A ideia é somar todos os valores dos pixels vizinhos dentro da janela deslizante do tamanho disponibilizado e depois dividir pelo número de pixels, fazendo assim a média local para dar o valor do pixel selecionado. 

b) Filtro separável
  Essa abordagem é tem a mesma ideia do primeiro, só que calculamos a média dos pixels em apenas um eixo (Janela horizontal), criando uma imagem borrada em apenas um eixo, para depois pegar essa imagem parcialmente borrada para fazer a média dos pixels no outro eixo (Janela vertical), gerando assim a imagem borrada da mesma maneira que o primeiro filtro, mas com menos cálculos.
  
  *Mais otimização: 
  Outra técnica que não conseguimos aplicar de maneira eficiente, mas que ajudaria ainda mais a deixar esse algoritmo mais rápido, seria implementar uma maneira de guardar as somas conforme a janela deslizante anda. 
  Supondo que estamos em uma janela 3x3, mas estamos calculando de maneira separável, então estamos vendo uma janela 1x3 que ficaria assim:
  
  (a b c) d e f  - Entre parênteses está a nossa janela
  
  Depois de fazer a média desta janela, andaríamos um pixel para o lado, ficando assim:
  
  a (b c d) e f  

  Se repararmos, calcular novamente a operação b + c seria refazer algo que já fizemos e em uma janela maior isso seria ainda mais evidente, já que calcularemos a soma de vários pixels repetidos. Dessa maneira, uma maneira mais inteligente seria somar apenas o novo pixel adicionado na janela e subtrair o pixel que saiu da janela, Ficando algo assim:
  
  Primeira média: soma = (a + b + c) e depois soma / 3
  Segunda média: soma = (a - soma + d) e depois soma / 3
  Terceira média: soma = (b - soma + e) e depois soma / 3
 
  **Isso seria aplicável também no primeiro filtro sem ser separável.


## c) Filtro Integral
  Essa abordagem seria a mais eficiente das 3. A ideia é criar uma imagem integral que seria a soma de todos os valores conforme a imagem avança, ficando algo como isso:
  
  Imagem original:
  (a) (b) (c)
  (d) (e) (f)
  (g) (h) (i)
  
  Imagem integral:
  (a)     (a+b)         (a+b+c)
  (a+d)   (a+b+d+e)     (a+b+c+d+e+f)
  (a+d+g) (a+b+d+e+g+h) (a+b+c+d+e+f+g+h+i)
  
  Com essa imagem integral fica mais simples e rápido de calcular a média. Em uma janela 3x3 a conta ficaria como:
  
  soma = img[i+1][j+1] - img[i+1][j-2] - img[i-2][j+1] + img[i-2][j-2]
  
  img[i][j] -> é o pixel central
  img[i+1][j+1] -> é o pixel da ponta inferior direita da janela
  img[i+1][j-2] -> é o pixel que fica fora da janela em cima do pixel da ponta superior direita da janela
  img[i-2][j+1] -> é o pixel que fica fora da janela ao lado do pixel da ponta inferior esquerda da janela
  img[i-2][j-2] -> é o pixel que fica fora da janela na diagonal do pixel da ponta superior esquerda
  
  
  
Para rodar o arquivo "main.py" que esta no diretório "Trabalho2-Blur", basta ter uma imagem "NOMEDAIMG.bmp" dentro da mesma pasta e com o nome correto no código, e depois usar o comando:

"$ python main.py"


3 - Bloom lighting: 
Para realizar esse trabalho foi utilizada a linguagem python e a biblioteca openCV.

4 - Desafio de segmentação: -

5 - Chroma key: -

6 - Trabalho final: -

