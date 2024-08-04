import io
# io modülü, Python'da giriş ve çıkış işlemleri (input/output) için kullanılır. Bu modül, metin ve ikili veri akışları ile çalışmak için çeşitli sınıflar ve fonksiyonlar sağlar.
import os 
# os modülü, işletim sistemi ile etkileşim kurmak için kullanılır. Dosya ve dizin işlemleri, çevre değişkenlerine erişim, işlem yönetimi gibi birçok fonksiyon sağlar
import unicodedata
import string
import glob
# glob modülü, dosya sisteminde belirli bir desene uyan dosyaları bulmak için kullanılır. Dosya yollarını belirli bir desenle eşleştirir. örneğin .txt olan tüm dosyaları bulabilir.
import torch
import random


# alphabet small + capital letters + ".,;"
# bir metnin tüm karakterlerini ( nokta , virgül ve noktalı virgül dahil) ayırır.
ALL_LETTERS = string.ascii_letters + ".,;"
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s) 
        # unicodedata.normalize fonksiyonu, s dizesini Unicode karakterlerini bileşenlerine ayırır. Örneğin, 'é' gibi aksanlı bir karakteri 'e' ve aksan olarak ayırır.
        if unicodedata.category(c) != 'Mn'
        # bu kod, e üzerindeki aksan işaretinin filtrelenmesini sağlar
        and c in ALL_LETTERS
    )

# Yani, örnekte c ifadesi sırayla her bir karakteri (örneğin c, a, f, e, ́) temsil eder. 
# Eğer bu karakter ALL_LETTERS içinde varsa ve kategori Mn değilse, bu karakter tutulur. café örneğinde, e aksanlı olduğu için e karakteri korunur, aksan işareti (́) filtrelenir. 
# Sonuçta, "café" dizesi "cafe" olarak döner.
# '.join(...)` ifadesi, bu karakterleri tek bir dize olarak birleştirir. Sonuç olarak, "café" dizesi "cafe" olarak döner.
# kısacası özel karakterleri atıyor, tüm karakterler ing haline geliyor.

def load_data():
    # build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)
    
    # read a file and split into lines
    # strip() metodu girilen(burası için boşluk) karakteri stringin başından ve sonundan siler
    # split('\n') metodu, metni her yeni satır karakterinde keser. 
    # io Modülü: Dosya açma ve okuma/yazma işlemleri için kullanılır.

    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
# os.path.splitext(os.path.basename(filename))[0]:
# os.path.splitext fonksiyonu, dosya adını ve uzantısını ayırır. Bu fonksiyonun çıktısı, bir demet (tuple) olup, birinci öğesi dosya adı, ikinci öğesi ise dosya uzantısıdır.
# [0] ile sadece dosya adını alır.
# Örneğin, english.txt dosyasını 'english' olarak ayırır ve category değişkenine atar.
        lines = read_lines(filename)
        category_lines[category] = lines
        # sözlüğün yapısı gereği anahtarlar category'ler oluyor. category'deki line lar da value lar oluyor.
    return category_lines, all_categories
    
"""
To represent a sinle letter, we use a 'one-hot vector' of 
size <1  * n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>

To make a word we join a bunch of those into a 
2D matrix <line_length * 1 * n_letters>.

1: PyTorch, verileri genellikle toplu (batch) olarak işler. 
Ancak burada sadece tek bir kelime üzerinde çalışıyoruz, bu yüzden batch boyutu 1'dir. Bu 1 boyutu, veriyi toplu olarak işlediğimizi belirtir.

That extra 1 dimension is because PyTorch assumes 
everthing is in batches - we're just using a batch size of 1 here.

Kelimeler ve Cümleler: Metin verileri, makine öğrenmesi ve derin öğrenme modelleri tarafından işlenebilmesi için sayısal formata dönüştürülmelidir. 
Tensörler, bu tür sayısal temsilleri temsil etmek için kullanılır.
 Örneğin, bir kelimenin tek-ısı (one-hot) vektörü, kelimenin sayısal bir temsilidir.
Gömme (Embedding): Kelimeler genellikle gömme vektörleri (word embeddings) kullanılarak temsil edilir. 
Bu vektörler, kelimeleri çok boyutlu bir uzayda yerleştirir ve benzer kelimeleri benzer vektörlerle temsil eder. Bu vektörler tensörler olarak saklanır.
"""

# Find letter index from all_letters, all_letters'ta kaçıncı indexte olduğunu bulur harfin ,e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 * n_letters> Tensor

def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

"""
1. torch.zeros(1, N_LETTERS)
Bu satır, 1 x N_LETTERS boyutunda, başlangıçta sıfırlarla doldurulmuş bir tensör oluşturur. N_LETTERS değişkeni, kullanılabilecek toplam karakter sayısını temsil eder. Örneğin, ALL_LETTERS içinde N_LETTERS kadar karakter varsa (örneğin 52 harf ve 3 noktalama işareti ile 55 karakter), tensör boyutu 1 x 55 olacaktır.

2. tensor[0][letter_to_index(letter)] = 1
Bu satır, verilen harfi (letter) tensörün doğru konumuna yerleştirir. letter_to_index(letter) fonksiyonu, harfin ALL_LETTERS içindeki konumunu (indeksini) döndürür. Bu indeks, tensörde 1 olarak ayarlanır ve diğer tüm değerler 0 kalır. Bu, one-hot encoding olarak bilinir ve her harfi benzersiz bir şekilde temsil eder.
"""


# turn a line into a <line_length * 1 * n_letters>,
# or an array of one-hot letter vectors 

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor
# enumerate(line) fonksiyonu, line içindeki her karakterin indeksini ve karakteri döndürür.


def random_training_example(category_lines, all_categories):
# random olarak bir name seçiyor ve karşılığı olduğu ülkeyi döndürüyor
    def random_choice(a):
        random_idx = random.randint(0, len(a)-1)
        return a[random_idx]  # rastgele bir indeks aldı
    
    category= random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype= torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor



if __name__ == '__main__':
    """
    Bu ifade, Python dosyasının doğrudan çalıştırıldığında (__main__ olarak) kodun bu kısmının yürütülmesini sağlar. 
    Bu, modül başka bir dosyadan import edildiğinde bu kodun çalışmamasını sağlar.
    """
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusarczyk'))

    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])

    print(letter_to_tensor('J')) #[1,57] 57 karakter ihtimali var büyük küçük bütün harfler
    print(line_to_tensor('Jones').size()) # [5,1,57]


