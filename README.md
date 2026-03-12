# VinBigData CXR Multi-Label Classification + Double Descent Study

## Projekt összefoglaló

Ez a projekt a **VinBigData mellkasröntgen (CXR) adathalmazon** végez **multi-label klasszifikációt** 5 kiválasztott kórképre, valamint a **double descent jelenség empirikus vizsgálatát** célozza kontrollált kísérletekben.

A fő cél egy olyan orvosi képfeldolgozó pipeline létrehozása, amely képszinten képes jelezni az alábbi különböző betegségeket.

A projekt második fókusza annak vizsgálata, hogyan alakul a generalizációs teljesítmény a modell kapacitásának, az adatmennyiségnek és bizonyos regularizációs/optimalizációs beállításoknak a függvényében.

## Célok és kimenetek

### Fő cél
5 finding képszintű multi-label osztályozása a VinBigData adathalmazon.

### Elvárt kimenetek
- képszintű predikciók, azaz 5 valószínűség képenként
- riportolt metrikák osztályonként és összesítve
- reprodukálható pipeline:
  - adat-előkészítés
  - split
  - tanítás
  - értékelés

## Adatkészlet

**Forrás:** VinBigData (Kaggle)

A projekt a következő adatokat használja:
- DICOM képek
- `train.csv` annotációs tábla

A `train.csv` sorai megfigyeléseket tartalmaznak, például:
- `image_id`
- `class_name`
- `class_id`
- `rad_id`
- bounding box koordináták: `x_min`, `y_min`, `x_max`, `y_max`

Fontos, hogy ugyanahhoz az `image_id`-hez több sor is tartozhat:
- több bbox miatt
- több radiológus annotációja miatt

Ezért a modelltanításhoz **képszintű aggregálás** szükséges.

## Label-kezelés

### Célfindingok
A projekt az alábbi 5 findingra fókuszál:
- Aortic enlargement
- Cardiomegaly
- Pleural thickening
- Pulmonary fibrosis
- Lung Opacity

### “No finding” kezelése
A `No finding` címkét nem külön tanítjuk, hanem a saját feladatdefiníciónk szerint vezetjük le:

- `No finding = 1`, ha az 5 célfinding mindegyike 0 képszinten
- `No finding = 0`, ha bármelyik célfinding 1

## Adattisztítás

A negatív példák konzisztens `No finding` jelentését biztosítani kell.

### Szűrési szabály
Kép-szinten töröljük azokat a képeket, ahol:
- az 5 célfinding közül egyik sincs jelen
- ugyanakkor van bármilyen más, bbox-szal jelölt, nem célzott abnormalitás

Ez azért szükséges, hogy a negatív példák ne tartalmazzanak olyan egyéb eltéréseket, amelyek a feladat szempontjából félrevezetők lennének.

## Radiológusi annotációk aggregálása

A `train.csv` sor-szintű annotációkat tartalmaz, ezért képszintű címkézés szükséges.

### Baseline aggregálási szabály
Egy adott célfinding képszintű címkéje:
- `1`, ha bármely radiológus jelölte
- `0`, ha egyik radiológus sem jelölte

Későbbi bővítési lehetőségek:
- majority vote
- soft label megközelítés

## Split stratégia

A split egysége az `image_id`.

Az `image_id` nem kerülhet egyszerre:
- train
- validation
- test

halmazba.

Vagyis a train és valid/test `image_id` halmazok metszete üres.

## Előfeldolgozás

A DICOM képeket konzisztens modellinputtá alakítjuk.

### Fő lépések
- DICOM beolvasás
- intenzitáskezelés konzisztens módon
- normalizálás
- egységes méretezés
- visszafogott augmentációk csak train halmazon

## Tanítás és értékelés

### Fő metrika
- per-class ROC-AUC
- ezek macro átlaga

### Később vizsgált lehetőségek
- súlyozás
- mintavételezés
- alternatív veszteségfüggvények

## Double Descent empirikus vizsgálat

A projekt külön célja a generalizációs teljesítmény vizsgálata a modell komplexitásának növelésével.

### Vizsgálati irányok

#### 1. Modellkomplexitás-sweep
Azonos adaton és tanítási recepten több, növekvő kapacitású modellváltozatot hasonlítunk össze.

Mérjük:
- train teljesítmény
- valid teljesítmény
- AUC alakulása a modellkapacitás függvényében

#### 2. Adatmennyiség-sweep
Fix modell mellett különböző train-méreteket vizsgálunk.

#### 3. Regularizáció / optimalizáció hatása
Kontrollált beállításváltoztatások vizsgálata, például:
- regularizáció erőssége
- augmentáció erőssége

### Reprodukálhatóság
- fix split
- fix seed
- néhány mérési ponton több seed-es ellenőrzés

## Benchmark és metrikai megjegyzés

A VinBigData Chest X-ray Kaggle versenyben az értékeléshez **IoU alapú mAP** metrikát használtak, ami objektumdetektálási feladatokra készült.

Mivel ebben a projektben **multi-label klasszifikációs feladatot** vizsgálunk, ezt a metrikát nem használjuk. Ehelyett klasszifikációs feladatokhoz megfelelő metrikákat alkalmazunk.

## Kapcsolódó benchmarkok

A projektterv két referenciát említ általános viszonyítási pontként:

### 1. VinDr-CXR klinikai rendszer
A dokumentum szerint egy saját fejlesztésű deep learning alapú rendszer, amely:
- multi-label klasszifikációt
- lokalizációt / detektálást

is tartalmaz.

A klasszifikációs rész példaként említett AUROC értékei:
- Pleural effusion: 98.9%
- Lung tumor: 97.8%
- Pneumonia: 96.9%
- Tuberculosis: 97.5%
- Other diseases: 92.0%

https://vinspace.edu.vn/bitstream/handle/VIN/179/An%20Accurate%20and%20Explainable%20Deep%20Learning%20System%20Improves%20Interobserver%20Agreement%20in%20the%20Interpretation%20of%20Chest%20Radiograph.pdf?isAllowed=y&sequence=1

### 2. Ismert mélytanulási modellek
A dokumentum említ egy collaborative deep learning modellt, a **VMNet**-et, amely több pre-trained architektúrát kombinál multi-label klasszifikációhoz.

Felsorolt modellek:
- ResNet50
- VGG19
- MobileNet
- VMNet

Példaként megadott ROC-AUC értékek:
- ResNet50: 73.6%
- VGG19: 86.4%
- MobileNet: 89.1%
- Ensemble (VMNet): 90.1%

https://www.sciencedirect.com/science/article/pii/S1877050925010981

## MLOps és reprodukálható pipeline

A projekt nemcsak a modellfejlesztésre fókuszál, hanem egy teljes, jól strukturált és reprodukálható ML pipeline kialakítására is.

### Fő MLOps elemek

#### Docker
A teljes futtatási környezet konténerizált, így a pipeline:
- lokálisan
- felhőben / HPC környezetben

is konzisztensen futtatható.

#### MLflow
Az MLflow a kísérletek kezelésére és modellverziózásra szolgál.

Nyomon követi:
- hyperparaméterek
- metrikák
- modellek
- artifactok

Lehetővé teszi a különböző futások összehasonlítását is.

#### DVC
A DVC az adatok és adatfeldolgozási pipeline verziókezelésére szolgál.

Célja:
- reprodukálhatóság javítása
- adatállapotok nyomon követése

#### FastAPI
A betanított modell inference kiszolgálásához FastAPI-t használunk.

A tervezett működés:
- a felhasználó képet küld az API-n keresztül
- a modell visszaadja az elváltozások valószínűségeit

#### Prefect
A Prefect a pipeline futtatásának:
- vezérlésére
- orkesztrálására
- automatizálására
- hibakezelésére
- ütemezésére

szolgál.

#### Prometheus + Grafana
A monitorozáshoz:
- **Prometheus** gyűjti a rendszer- és alkalmazásmetrikákat
- **Grafana** dashboardokon jeleníti meg azokat

Tipikus monitorozott mutatók:
- API hívások száma
- válaszidők
- CPU / memóriahasználat
- hibaarány

## Infrastrukturális háttér

A projektterv alapján a számításigényes modelltanítás:
- felhőben / HPC környezetben
- GPU erőforrásokon
- a **Komondor HPC infrastruktúrán**

történik.