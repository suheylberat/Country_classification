import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example


"""
nn.Module, PyTorch'un temel yapı taşı olan bir sınıftır ve tüm sinir ağı modüllerinin (katmanlar, modeller vb.) türetildiği temel sınıftır. 
Bir sinir ağı modelini oluştururken nn.Module sınıfından miras almak, modelin PyTorch'un sağladığı özelliklerden faydalanmasını sağlar.
PyTorch, derin öğrenme ve sinir ağı modelleri oluşturmak için kullanılan bir kütüphanedir.
Parametre Yönetimi:

nn.Module sınıfından miras aldığınızda, tanımladığınız her katman (örneğin nn.Linear, nn.Conv2d) otomatik olarak parametrelerini (ağırlıklar ve biaslar) yönetir. Bu, model parametrelerine kolayca erişmenizi ve optimizasyon işlemlerini gerçekleştirmenizi sağlar.
Hesaplama Grafiği:

nn.Module, ileri (forward) ve geri (backward) geçişleri otomatik olarak tanımlar ve hesaplama grafiğini oluşturur. Bu, geriye yayılım (backpropagation) ve ağırlık güncelleme işlemlerini kolaylaştırır.

"""
class RNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        """super(RNN, self).__init__() ifadesi, RNN sınıfının nn.Module sınıfından miras aldığı __init__() yöntemini çağırır.
        nn.Module sınıfının __init__() yöntemi, modelin temel yapı taşlarını kurar. Bu, parametrelerin ve alt modüllerin doğru bir şekilde kaydedilmesini içerir."""

        # we want to store our hidden size. Sınıfın bir özelliiği olarak saklanmasını sağladık hidden_size'ın. Bu, modelin diğer bölümlerinde bu boyutun kullanılmasını sağlar.
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # "+" because we combine them 
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # giriş boyutu artılı olan kısım, virgülden sonrası çıkış boyutu
        # we also need a softmax layer
        self.softmax = nn.LogSoftmax(dim=1)
        # dim=1 ifadesi, Softmax işleminin satırlar boyunca, yani her satır için ayrı ayrı uygulanacağını belirtir. Bu, her satırın kendi olasılık dağılımını elde etmesini sağlar.
        # nn.LogSoftmax(dim=1) uygulandığında, her satır Softmax işlemine tabi tutulur ve sonuç logaritmik olasılık olarak döner:
        
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor,hidden_tensor), 1)
        # dim=1, tensorlerin sütunlar (yani, ikinci boyut) boyunca birleştirileceği anlamına gelir.
        # dim=1 kullanarak iki tensorü birleştiriyorsanız, tensorlerin birinci boyuttaki (satır sayısı) değerleri aynı olmalıdır, ancak ikinci boyuttaki (sütun sayısı) değerler farklı olabilir ve bu değerler birleştirilir.
       

       #burayı anlamadım
        hidden = self.i2h(combined) # bu katman combined tensorünü ağırlık matrisi ile çarpar ve bias ekler, yeni bir hidden tensorü üretir
        output = self.i2o(combined) 
        output = self.softmax(output)
        """
        İlk satır (self.softmax = nn.LogSoftmax(dim=1)) bir LogSoftmax katmanı oluşturuyor ve bunu self.softmax olarak sınıfın bir özniteliği yapıyor.
        İkinci satır (output = self.softmax(output)) bu oluşturulan katmanı kullanarak output üzerinde işlem yapıyor.
        """
        return output, hidden


    # helper func but check out again
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    """Bu kodda, torch.zeros(1, self.hidden_size) ifadesi, boyutu 1 x hidden_size olan ve tüm elemanları sıfır olan bir tensor oluşturur. 
    Bu tensor, modelin hidden stateinin başlangıç değerini temsil eder."""
    
# category_lines is a dict with the country as key and the names as values and this is just a list of all the different countries  
category_lines, all_categories = load_data()
n_categories = len(all_categories)
print(n_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# one single step
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden() # RNN class'ındaki init_hidden'a götürüyor.


# bu kısımdaki rnn bir fonksiyon gibi çağırılmış ve input_tensor ve hidden_tensor argüman olarak geçirir. Bu modelin forward metodunu tetikler.
output, next_hidden = rnn(input_tensor, hidden_tensor)
print(output.size())
print(next_hidden.size())




# whole sequence/name
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()


# bu kısımdaki rnn bir fonksiyon gibi çağırılmış ve input_tensor ve hidden_tensor argüman olarak geçirir. Bu modelin forward metodunu tetikler.
output, next_hidden = rnn(input_tensor[0], hidden_tensor) # RNN, önceki harflerin bilgisini "hidden state" aracılığıyla sonraki adımlara taşır.
# RNN her harf için bir çıktı üretir, ama biz genellikle son harfin çıktısıyla ilgileniriz.

print(output.size())
print(next_hidden.size())

# 
def category_from_output(output):
    category_idx = torch.argmax(output).item() # argmax sayesinde ismin ait olma ihtimali en yüksek dili buluyoruz ancak bu indeksi bir PyTorch tensörü olarak değil, bir Python skaler değeri olmasını istiyoruz bunun için de item metodunu kullanıyoruz. 
    return all_categories[category_idx]

print(category_from_output(output))

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(),lr= learning_rate) # Stochastic gradient descent ,, learning rate defined as learning_rate
# ağırlıkları güncellemek için kullanılan optimizasyon algoritması. sgd

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden() # we get the intial hidden state

# torch.Tensor.size() fonksiyonu, bir tensörün boyutlarını (dimensions) bir tuple olarak döndürür. 
# Bu, tensörün her boyutundaki eleman sayısını gösterir. 

    # category_tensor aslında bu ismin gerçek kategorisini temsil eder
    for i in range(line_tensor.size()[0]): # the length of the name basically
        output, hidden = rnn(line_tensor[i], hidden) # rnn'deki forward pass'e gönderilir değerler ve bir sonuç alınır

    # buradaki output ise rnn'in tahmin ettiği kategori olasılığı
    loss = criterion(output, category_tensor) # tahmin olan kategori(output) ile gerçek kategori(category_tensor) arasındaki fark hesaplanır.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 120000

"""
Özetle, 100000 iterasyon, modelin 100000 kez bir isim üzerinde çalışacağı anlamına gelir, ancak bu 100000 farklı isim demek değildir. 
Model, sınırlı sayıdaki ismi tekrar tekrar görerek öğrenir ve genelleme yapmayı öğrenir. 
Bu, gerçek dünya uygulamalarında yaygın bir eğitim yaklaşımıdır.
"""
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)


    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps) # every thousand stepte all_losses a ekliyoruz sonra current_losses'ı sıfır yapıyoruz. plot_steps' e bölüyoruz ortalamayı almak için
        current_loss = 0


    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG({category})"
        print(f"{i} {i/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

""" {i}, kaçıncı eğitim adımında olduğumuzu gösteriyor.
    {i/n_iters*100}, eğitim tamamlanma yüzdesi
    {loss:.4f}, mevcut iterasyondaki kayıp(loss) değeri gösteriyor. Düşük kayıp değerleri, modelin daha iyi performans gösterdiğini işaret eder.
    .4f ise kaybın 4 ondalık basamakla gösterilmesini sağlar.
    Kayıp, modelin tahminleri ile gerçek değerler arasındaki farkı ölçen bir sayısal değerdir.
    {line}, modelin tahmin yapmaya çalıştığı ismi gösterir.
    {guess}, modelin tahminini yani tahmin edilen ülkeyi gösterir
    {correct}, tahminin doğru olup olmadığını gösterir.
"""



# loss'u matplotlib ile görselleştiriyor.
plt.figure()
plt.plot(all_losses)
plt.show() 


def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
        print(guess)

while True:
    sentence = input("Input:")
    if sentence == "quit":
        break

    predict(sentence)

"""

1. `category_from_output` fonksiyonu:
   - Bu fonksiyon bir ülke (veya milliyet) döndürür, isim değil.
   - `all_categories` listesi muhtemelen ülke veya milliyet isimlerini içerir (örneğin: "İtalyan", "Alman", "Japon" gibi).

2. `output, hidden = rnn(line_tensor[i], hidden)` ifadesi:
   - `output`: Bu, RNN'in her bir harf için ürettiği çıktıdır. Doğrudan bir ülke veya şehir değildir.
   - Bu çıktı, her kategori (ülke/milliyet) için bir olasılık dağılımını temsil eder.
   - Örneğin, eğer 10 farklı ülke kategorisi varsa, `output` 10 elemanlı bir vektör olabilir, her eleman bir ülkenin olasılığını gösterir.

3. `guess = category_from_output(output)` ifadesi:
   - Burada `output` fonksiyona girdi olarak verilir çünkü:
     - `output`, RNN'in son tahminini içerir (ismin son harfi işlendikten sonra).
     - `category_from_output` fonksiyonu, bu olasılık dağılımından en yüksek olasılığa sahip kategoriyi (ülkeyi) seçer.

Örnek olarak açıklayalım:

- Diyelim ki bir "Mario" ismini işliyoruz.
- RNN her harf için bir çıktı üretir, ama biz genellikle son harfin çıktısıyla ilgileniriz.
- Son `output` şöyle görünebilir: [0.1, 0.7, 0.05, 0.15] (basitleştirilmiş örnek)
  - Bu, sırasıyla [Alman, İtalyan, Japon, Fransız] olasılıklarını temsil edebilir.
- `category_from_output(output)` fonksiyonu bu vektördeki en yüksek değeri bulur (0.7) ve buna karşılık gelen kategoriyi ("İtalyan") döndürür.

Özetle:
- `category_from_output` bir ülke/milliyet döndürür.
- `output` doğrudan bir ülke değil, ülke olasılıklarının bir vektörüdür.
- `category_from_output(output)` kullanılır çünkü `output`'tan en olası ülkeyi seçmemiz gerekir.

Bu yapı, modelin her bir isim için çeşitli ülke olasılıklarını hesaplamasına ve sonra en olası ülkeyi tahmin etmesine olanak sağlar.

"""

"""
Özetle, model önce tamamen eğitilir (100000 iterasyon), sonra eğitim bittiğinde kullanıcıdan girdi almaya başlar. 
Bu aşamada model artık yeni bir şey öğrenmez, sadece eğitim sırasında öğrendiklerini kullanarak tahminlerde bulunur. 
Bu yaklaşım, modelin önce kapsamlı bir şekilde öğrenmesini 
ve sonra öğrendiklerini pratik bir şekilde uygulamasını sağlar.
"""


"""
Çok iyi bir soru. RNN'in (Tekrarlayan Sinir Ağı) isim-milliyet ilişkisini nasıl öğrendiğini ve kullandığını daha detaylı açıklayalım:

1. Karakter Seviyesinde Öğrenme:
   - RNN, isimleri harf harf işler.
   - Her harf için, önceki harflerin bağlamını da dikkate alır.

2. İstatistiksel Örüntüler:
   - Model, eğitim verilerindeki istatistiksel örüntüleri öğrenir.
   - Örneğin, "-elli" son ekinin İtalyan isimlerinde sık görüldüğünü öğrenebilir.

3. Karakter Dizileri:
   - Sadece tek harfleri değil, harf dizilerini de öğrenir.
   - "Mar-" başlangıcının İtalyan isimlerinde yaygın olduğunu fark edebilir.

4. Bağlamsal Öğrenme:
   - Bir harfin kendisinden ziyade, diğer harflerle ilişkisini öğrenir.
   - "Mario" için, "M-a-r-i-o" dizisinin bütününü değerlendirir.

5. Olasılık Dağılımları:
   - Her milliyet için bir olasılık dağılımı oluşturur.
   - "Mario" için İtalyan olasılığı yüksek, Japon olasılığı düşük olabilir.

6. Genelleme Yeteneği:
   - Daha önce görmediği isimlere de uygulanabilir örüntüler öğrenir.
   - Örneğin, "Luca" ismini daha önce görmemiş olsa bile, İtalyan olarak tahmin edebilir.

7. Çoklu Özellik Kombinasyonu:
   - Sadece tek bir özelliğe (örneğin son ek) değil, birçok özelliğin kombinasyonuna bakar.
   - Başlangıç harfleri, orta kısım, son ekler gibi farklı bölümleri birlikte değerlendirir.

8. Sıralı Bağımlılıklar:
   - Harflerin sırasının önemini öğrenir.
   - "Mario" ve "Oimar" farklı şekillerde değerlendirilir.

9. Negatif Örnekler:
   - Bir ismin hangi milliyete ait olmadığını da öğrenir.
   - "Mario"nun Çince bir isim olmadığını anlar.

10. Sürekli Güncelleme:
    - Her eğitim örneğiyle bilgisini günceller ve ince ayar yapar.

Özetle, model basit "eğer-o zaman" kuralları kullanmaz. Bunun yerine, karmaşık ve çok boyutlu bir olasılık uzayında çalışır. Her harf ve harf dizisi için, farklı milliyetlere ait olma olasılıklarını hesaplar ve bu olasılıkları sürekli günceller. Sonuç olarak, "Mario" ismini gördüğünde, tüm bu öğrenilmiş örüntüleri ve istatistikleri kullanarak İtalyan olma olasılığını yüksek hesaplar.

"""

"""
periodic_retraining fonksiyonu, bellek havuzundaki tüm örnekler üzerinde modeli yeniden eğitir.
Ana döngüde, her 10 iterasyonda bir periodic_retraining fonksiyonu çağrılır.
"""