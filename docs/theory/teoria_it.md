# Double descent nei modelli lineari: teoria (Assignment 4, SMML 2025/26)

> Versione 4, 25/10/2026. Versione con formule impaginate: `teoria_it.pdf`. English version: `theory_en.md`.

#### Come usare questo documento.

Leggilo in circa 40 minuti. Le sezioni 2–6 sono la teoria minima da saper spiegare all’orale. La sezione 7 collega la teoria all’estensione con la gradient descent, la sezione 8 descrive cosa ci aspettiamo di vedere nei grafici, la sezione 9 il disegno degli esperimenti. In fondo: domande d’orale con risposta e soluzioni delle domande di verifica. Per ogni affermazione indico la fonte; quando un concetto **non** è nei materiali del corso del corso (slide 1–15 di Cesa-Bianchi) lo segnalo esplicitamente come fonte esterna.

## Notazione e corrispondenze

<div class="center">

| Concetto                        | Corso (slide 9)                | Assignment / paper             | Codice  |
|:--------------------------------|:-------------------------------|:-------------------------------|:--------|
| numero di esempi di training    | $m$                            | $n$                            | `m`     |
| numero di feature (complessità) | $d$                            | $d$ (assignment) / $N$ (paper) | `d`     |
| design matrix                   | $X \in \mathbb{R}^{m\times d}$ |                                | `X`     |
| etichette                       | $y \in \mathbb{R}^m$           |                                | `y`     |
| parametro ridge                 | $\alpha$                       | $\lambda$ (altri testi)        | `alpha` |
| soglia di interpolazione        | $d = m$                        | $N = n$                        |         |

</div>

**Attenzione:** nel paper $d$ è la dimensione dell’*input* $x$, mentre il numero di feature/parametri è $N$ e i campioni sono $n$ (p. 2 e p. 4).

Nel report scriveremo una volta sola: “seguiamo la notazione del corso; il numero di esempi $n$ dell’assignment corrisponde a $m$”. Le righe di $X$ sono gli esempi: $X^\top = [x_1,\dots,x_m]$ (slide 9, p. 5). Un predittore lineare per la regressione è $h(x) = w^\top x$ con $w \in \mathbb{R}^d$ (slide 9, p. 1 e p. 4). L’intercetta si gestisce aggiungendo una feature costante pari a $1$ (slide 9, p. 1) oppure centrando i dati.

## Setup statistico: rischio, errore di training, errore di test

*Fonte: slide “4. statRisk” (p. 1–4) e “1. intro” (p. 3–5).*

- Ogni esempio $(x,y)$ è estratto in modo indipendente da una distribuzione $\mathcal{D}$ fissata ma sconosciuta (statRisk, p. 1). Training set e test set sono quindi campioni statistici.

- **Rischio statistico** di un predittore $h$ rispetto alla loss $\ell$ (statRisk, p. 1): $\ell_{\mathcal{D}}(h) = \mathbb{E}\bigl[\ell(h(X), Y)\bigr]$. Per noi $\ell(\hat y, y) = (\hat y - y)^2$ e $h(x) = w^\top x$.

- **Predittore Bayes-ottimo** $f^\ast$: il migliore possibile data $\mathcal{D}$. Per la square loss $f^\ast(x) = \mathbb{E}[Y \mid X = x]$ e il **rischio di Bayes** è la varianza condizionata attesa dell’etichetta (statRisk, p. 1–2). Nel nostro dataset sintetico, con $y = w^{\star\top}x + \varepsilon$ e rumore di varianza $\sigma^2$, il rischio di Bayes è proprio $\sigma^2$: **nessun modello può avere test error atteso sotto $\sigma^2$**. Lo useremo come linea di riferimento nei grafici.

- **Errore di training**: $\ell_S(h) = \frac{1}{m}\sum_{t=1}^{m} \ell(h(x_t), y_t)$; per noi $\ell_S(w) = \frac{1}{m}\left\lVert Xw - y \right\rVert^2$. ERM sceglie il predittore che lo minimizza (intro, p. 4).

- **Errore di test**: la stessa media calcolata su un test set $S'$ indipendente; stima il rischio $\ell_{\mathcal{D}}(h)$, e per Chernoff–Hoeffding la stima è accurata quando il test set è grande (statRisk, p. 3–4). Il test set non va mai usato per addestrare né per scegliere iperparametri.

#### Overfitting e underfitting nel corso.

Intro, p. 5: se l’algoritmo restituisce un predittore con training error alto è *underfitting*; con training error basso ma test error alto è *overfitting*. Il corso osserva che ci si aspetta overfitting quando la classe di predittori è troppo ricca rispetto a $m$ (“$\log_2|F| \gg m$”, intro p. 5). **Il double descent è esattamente un caso in cui questa intuizione classica non basta**: per $d \gg m$ il training error è $0$ ma il test error torna a scendere.

## Least squares con $d \le m$: la formula delle slide

*Fonte: slide 9, p. 4–5.*

Il predittore di regressione lineare è l’ERM rispetto alla square loss: $$w_S = \operatorname*{argmin}_{w \in \mathbb{R}^d} \left\lVert Xw - y \right\rVert^2 .$$ $F(w) = \left\lVert Xw - y \right\rVert^2$ è convessa, quindi il minimo si trova dove il gradiente è nullo: $$\nabla F(w) = 2X^\top(Xw - y) = 0
  \quad\Longleftrightarrow\quad
  X^\top X\, w = X^\top y \qquad \text{(equazioni normali)}.$$ Se $X^\top X$ è invertibile: $$\boxed{w_S = (X^\top X)^{-1} X^\top y}$$ **Quando è invertibile?** La slide dice: se e solo se $x_1,\dots,x_m$ generano $\mathbb{R}^d$. Servono almeno $d$ vettori per generare $\mathbb{R}^d$, quindi serve $m \ge d$.

#### Interpretazione geometrica.

Il vettore delle predizioni sul training set è $Xw_S = X(X^\top X)^{-1}X^\top y = Py$, dove $P$ è la proiezione ortogonale sullo spazio generato dalle colonne di $X$. Più feature aggiungiamo, più grande è quello spazio, più $Py$ si avvicina a $y$: l’errore di training **non può aumentare** quando $d$ cresce.

#### Il caso $d = m$.

Se $X$ è quadrata e invertibile, $w_S = X^{-1}y$ e $Xw_S = y$: errore di training esattamente $0$. Il modello **interpola** già a $d = m$. Da qui il nome “soglia di interpolazione”.

## Least squares con $d > m$: la soluzione a minima norma

*Fonte esterna (non nei materiali del corso): pseudo-inversa di Moore–Penrose, testi standard di algebra lineare (es. Strang, *Introduction to Linear Algebra*, capitolo su pseudo-inversa e SVD); uso della soluzione a minima norma: Belkin et al. 2019, sez. 2 (p. 4: “quando il minimizzatore non è unico … scegliamo quello con norma $\ell_2$ minima”) e p. 10 (“minimum norm linear regression in the feature space”).*

## Perché la formula delle slide non funziona più

$X^\top X$ è una matrice $d \times d$, ma il suo rango è uguale al rango di $X$, che è al massimo $\min(m,d) = m$. Se $d > m$ allora $\operatorname{rank}(X^\top X) \le m < d$: la matrice è **singolare** e $(X^\top X)^{-1}$ non esiste. Equivalentemente: $m$ vettori non possono generare $\mathbb{R}^d$ se $m < d$.

## Infinite soluzioni che interpolano

Con $d > m$ (e righe di $X$ linearmente indipendenti, il caso tipico con dati continui), il sistema $Xw = y$ ha $m$ equazioni e $d$ incognite: ha **infinite** soluzioni esatte. Se $w_0$ è una soluzione, lo è anche $w_0 + z$ per ogni $z$ nel nucleo di $X$ ($Xz = 0$), che ha dimensione $d - m$. Tutte hanno errore di training $0$.

## Quale scegliere: la soluzione a minima norma

$$\boxed{w^{+} = \operatorname*{argmin}\bigl\{\left\lVert w \right\rVert : Xw = y\bigr\} = X^\top (X X^\top)^{-1} y}$$ **Verifica che interpola:** $Xw^{+} = XX^\top(XX^\top)^{-1}y = y$. ($XX^\top$ è $m\times m$ e invertibile se le righe di $X$ sono indipendenti.)

**Perché ha norma minima (dimostrazione breve, da saper fare).** $w^{+} = X^\top a$ con $a = (XX^\top)^{-1}y$, quindi $w^{+}$ è combinazione lineare delle righe di $X$. Ogni altra soluzione è $w = w^{+} + z$ con $Xz = 0$, cioè $z$ è ortogonale a tutte le righe di $X$, e quindi a $w^{+}$. Per Pitagora: $$\left\lVert w \right\rVert^2 = \left\lVert w^{+} \right\rVert^2 + \left\lVert z \right\rVert^2 \ge \left\lVert w^{+} \right\rVert^2 .$$

## Una formula unica: la pseudo-inversa

La pseudo-inversa $X^{+}$ si calcola dalla SVD $X = U\Sigma V^\top$: $X^{+} = V\Sigma^{+}U^\top$, dove $\Sigma^{+}$ inverte i valori singolari non nulli e lascia a zero gli altri. Vale:

- se $d \le m$ e $X$ ha rango pieno: $X^{+}y = (X^\top X)^{-1}X^\top y = w_S$ (la formula delle slide);

- se $d > m$ e $X$ ha rango $m$: $X^{+}y = X^\top(XX^\top)^{-1}y = w^{+}$.

Quindi in numpy basta `w = np.linalg.pinv(X) @ y` per **tutti** i $d$. È algebra lineare, non un “fit” di libreria, quindi rispetta il vincolo “from scratch” (FAQ dell’assignment, p. 16).

#### Note numeriche (importanti per il codice).

- Non usare `np.linalg.inv(X.T @ X)`: formare $X^\top X$ eleva al quadrato il numero di condizionamento, e vicino a $d=m$ i risultati diventano numericamente inaffidabili.

- `np.linalg.pinv` azzera solo i valori singolari sotto una soglia minuscola (`rcond`, circa $10^{-15}$ relativo al più grande). Vicino a $d=m$ i valori singolari piccoli *non* vengono azzerati: `pinv` non regolarizza, quindi il picco resta visibile. È il comportamento che vogliamo mostrare.

## Esempio numerico minimo

$m=1$, $d=2$: un solo esempio $x = (1,1)$, $y = 2$. Le soluzioni di $w_1 + w_2 = 2$ sono infinite: $(2,0)$, $(0,2)$, $(1,1)$, $(3,-1)$… La soluzione a minima norma è $w^{+} = x^\top(xx^\top)^{-1}y = (1,1)\cdot\frac{2}{2} = (1,1)$, con norma $\sqrt{2}$, contro norma $2$ di $(2,0)$. Distribuisce il peso in modo uniforme invece di “puntare tutto” su una feature.

## Perché il test error esplode a $d \approx m$

*Fonte: slide 9, p. 5: “quando $X^\top X$ è quasi singolare, $w_S$ è molto sensibile a perturbazioni del training set. Questa instabilità aumenta l’errore di stima (varianza)”. È l’aggancio al corso.*

## Intuizione con un esempio $2\times 2$

$m = d = 2$, righe $x_1 = (1,0)$, $x_2 = (1,\varepsilon)$ con $\varepsilon$ piccolo (righe quasi parallele, $X$ quasi singolare). Etichette $y = (1,\, 1+\delta)$, dove $\delta$ è un piccolo rumore. Risolvendo $Xw = y$: $w_1 = 1$ e $w_1 + \varepsilon w_2 = 1+\delta$, quindi $$w_2 = \delta/\varepsilon .$$ Con $\varepsilon = 0.01$ e un rumore $\delta = 0.1$, otteniamo $w_2 = 10$: un rumore piccolo produce un peso enorme. Su un nuovo punto, ad esempio $x = (0,1)$, la predizione vale $10$ invece di circa $0$. È esattamente il picco.

## Versione generale con gli autovalori

Siano $\lambda_1 \ge \dots \ge \lambda_d \ge 0$ gli autovalori di $X^\top X$ e $v_1,\dots,v_d$ gli autovettori (ortonormali). La soluzione a minima norma si scrive $$w^{+} = \sum_{i:\,\lambda_i > 0} \frac{v_i^\top X^\top y}{\lambda_i}\, v_i .$$ Supponiamo label lineari con rumore: $y = Xw^\star + \varepsilon$. La parte dovuta al rumore è $\sum_i \frac{v_i^\top X^\top \varepsilon}{\lambda_i} v_i$; ogni termine ha norma $\frac{|u_i^\top \varepsilon|}{\sqrt{\lambda_i}}$, con $u_i = Xv_i/\sqrt{\lambda_i}$ vettore unitario. **Il rumore viene diviso per $\sqrt{\lambda_i}$.**

- $d \ll m$: $X$ è “alta e stretta”, tutti gli autovalori sono grandi (circa $m$ per feature standardizzate): il rumore viene smorzato.

- $d \approx m$: $X$ è quasi quadrata e casuale, il più piccolo autovalore è vicinissimo a $0$: il rumore viene amplificato, $\left\lVert w \right\rVert$ esplode, e con lui il test error.

- $d \gg m$: $X$ è “larga”, gli $m$ autovalori non nulli tornano lontani da $0$ e il rumore viene “spalmato” su molte direzioni: $\left\lVert w^{+} \right\rVert$ e il test error scendono. È la **seconda discesa**.

*Fonte esterna (non nei materiali del corso): il comportamento del più piccolo autovalore di una matrice casuale in funzione di $d/m$ è un risultato di random matrix theory (legge di Marchenko–Pastur). Non lo dimostriamo: lo mostriamo empiricamente plottando $\left\lVert w \right\rVert$ in funzione di $d$, che è più difendibile all’orale.*

## Una formula esatta per il regime classico (facoltativa)

*Fonte esterna (non nei materiali del corso): Hastie, Montanari, Rosset, Tibshirani, “Surprises in high-dimensional ridgeless least squares interpolation”, Annals of Statistics 2022.*

Se $x \sim \mathcal{N}(0, I_d)$, $y = w^{\star\top}x + \varepsilon$ con rumore di varianza $\sigma^2$ e $d < m-1$, allora $$\mathbb{E}\bigl[\left\lVert w_S - w^\star \right\rVert^2\bigr] = \sigma^2\,\frac{d}{m - d - 1},$$ che **diverge per $d \to m-1$**. Non serve dimostrarla; è utile per dire all’orale che il picco non è un artefatto numerico ma un fenomeno statistico previsto dalla teoria.

## Ridge regression

*Fonte: slide 9, p. 5.*

## Formula

$$w_{S,\alpha} = \operatorname*{argmin}_{w \in \mathbb{R}^d} \left\lVert Xw - y \right\rVert^2 + \alpha \left\lVert w \right\rVert^2 ,
  \qquad \alpha > 0 .$$ Gradiente: $2X^\top(Xw - y) + 2\alpha w = 0 \;\Rightarrow\; (X^\top X + \alpha I)\,w = X^\top y$, quindi $$\boxed{w_{S,\alpha} = (\alpha I + X^\top X)^{-1} X^\top y}$$

## Perché elimina il picco

Gli autovalori di $\alpha I + X^\top X$ sono $\alpha + \lambda_i \ge \alpha > 0$ (slide 9, p. 5): la matrice è sempre invertibile, per ogni $d$, anche $d > m$. Nella base degli autovettori: $$w_{S,\alpha} = \sum_{i} \frac{v_i^\top X^\top y}{\lambda_i + \alpha}\, v_i .$$ Si divide per $\lambda_i + \alpha$ invece che per $\lambda_i$: anche quando $\lambda_i \approx 0$ il denominatore resta almeno $\alpha$, quindi il rumore non può essere amplificato senza limite. Il picco si attenua o sparisce.

## Il prezzo: bias

- $\alpha \to 0$: si ritrova la soluzione a minima norma $w^{+}$ (anche per $d > m$; nelle direzioni con $\lambda_i = 0$ il numeratore $v_i^\top X^\top y$ è $0$). *Fatto standard, non nelle slide.*

- $\alpha \to \infty$: $w_{S,\alpha} \to 0$, il predittore ignora i dati (slide 9, p. 5: $\alpha$ controlla il bias).

- In mezzo: $\alpha$ scambia varianza (instabilità) con bias (sotto-adattamento). Con $\alpha$ troppo grande il test error sale *per tutti* i $d$.

*Fonte: Belkin et al. 2019, p. 10: la regolarizzazione “può sia impedire l’interpolazione sia cambiare la capacità effettiva della classe, attenuando o mascherando il picco”.*

*Fonte esterna (non nei materiali del corso): Nakkiran, Venkat, Kakade, Ma, “Optimal regularization can mitigate double descent”, ICLR 2021: con $\alpha$ ottimale la curva del test error diventa monotona.*

## Calcolo in pratica

- Primale: `np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ y)` (sistema $d \times d$).

- Duale, più economico per $d > m$: $w_{S,\alpha} = X^\top(XX^\top + \alpha I)^{-1}y$ (sistema $m\times m$). Le due forme coincidono perché $X^\top(XX^\top + \alpha I) = (X^\top X + \alpha I)X^\top$.

- `solve` è preferibile a calcolare l’inversa esplicita (più stabile e più veloce).

## Come scegliere $\alpha$ senza data leakage

- Per il grafico principale: alcune curve con $\alpha$ fissi (es. $10^{-3}, 10^{-1}, 10$) accanto alla curva di minima norma. Nessuna scelta sui dati, nessun leakage.

- Se avanza tempo: $\alpha$ scelto per ogni $d$ con un validation split (o $K$-fold cross-validation) ricavato **solo** dal training set (slide “6. crossVal”, p. 2–3: tuning su validation set; p. 3–4: nested CV), come nel notebook `CrossValidation.ipynb` del prof. Il test set non si usa mai per scegliere.

## Collegamento con la gradient descent (estensione)

*Fonte: slide 10 (online gradient descent) per l’algoritmo; il risultato sulla minima norma è esterno (vedi sotto).*

Gradient descent su $F(w) = \left\lVert Xw - y \right\rVert^2$ partendo da $w_1 = 0$: $$w_{k+1} = w_k - \eta\, 2X^\top(Xw_k - y).$$

- Ogni aggiornamento è $X^\top(\dots)$, cioè una combinazione lineare delle righe di $X$. Partendo da $0$, $w_k$ resta sempre nello spazio delle righe.

- Con passo $\eta$ abbastanza piccolo ($\eta < 1/\lambda_1$), per $d > m$ la GD converge a una soluzione che interpola *e* sta nello spazio delle righe: per la dimostrazione della Sezione 4.3, è proprio $w^{+}$.

- Fermarsi prima della convergenza (early stopping) ha un effetto simile alla ridge: le direzioni con $\lambda_i$ piccolo sono le ultime a essere apprese, quindi il rumore lungo di esse non viene amplificato.

*Fonte: bias implicito della GD verso la minima norma: Belkin et al. 2019, p. 11 (“SGD initialized at zero also converges to the minimum norm solution”).*

L’estensione si fa **solo** se report e repo sono completi.

## Cosa ci aspettiamo nei grafici

## La curva a U classica e il double descent

- **Regime classico** ($d < m$): il corso scompone il rischio di $h_S$ in *errore di stima* (varianza) $+$ *errore di approssimazione* (bias) $+$ rischio di Bayes (statRisk, p. 4). Aggiungendo feature il bias scende ma la varianza sale; overfitting quando domina la varianza, underfitting quando domina il bias (statRisk, p. 6). Vicino a $d = m$ la varianza va a infinito: è la curva a U.

- **Soglia di interpolazione** ($d = m$): l’errore di training arriva a $0$, il test error ha il picco.

- **Regime sovra-parametrizzato** ($d > m$): errore di training sempre $0$, il test error scende di nuovo e può andare sotto il minimo del regime classico (paper, Fig. 1, p. 3). Spiegazione del paper (p. 3 e p. 6): oltre la soglia, classi più ricche contengono interpolanti di norma più piccola, cioè “più semplici” (rasoio di Occam); la norma è massima alla soglia e poi decresce (Fig. 2, p. 5). La teoria classica, che prevede solo la U, non spiega questa parte: è il punto del paper.

## Grafici previsti

1.  Errore di training e di test in funzione di $d$ (minima norma), linea verticale a $d = m$, asse $y$ logaritmico (il picco è di ordini di grandezza).

2.  Sul sintetico, linea orizzontale al rischio di Bayes $\sigma^2$ (Sezione 2) come riferimento.

3.  $\left\lVert w \right\rVert$ in funzione di $d$: deve avere il picco nello stesso punto (evidenza del meccanismo della Sezione 5).

4.  Test error di minima norma e ridge (più valori di $\alpha$) sullo stesso grafico.

5.  Gli stessi grafici per il dataset reale, e un confronto tra i due dataset.

## Disegno degli esperimenti (anteprima)

#### Rapporto con il paper.

Il paper mostra il fenomeno con random Fourier features e random ReLU features (sez. 2 e App. C.1–C.2), reti neurali (sez. 2) e foreste/boosting (sez. 3), su MNIST e altri dataset di classificazione. Non contiene un esperimento di regressione lineare “pura”, ma le random features sono addestrate con regressione lineare a minima norma nello spazio delle feature (p. 10): il nostro esperimento ne è la versione più semplice.

## Dataset sintetico: label lineari rumorose

*Fonte esterna (non nei materiali del corso): modello a “feature deboli”: Belkin, Hsu, Xu, “Two models of double descent for weak features”, 2019.*

- $x \sim \mathcal{N}(0, I_D)$ con $D$ grande (es. $D = 1000$), $y = w^{\star\top}x + \varepsilon$, rumore gaussiano.

- Il modello con complessità $d$ usa solo le prime $d$ feature. Per $d < D$ le feature non usate agiscono come rumore aggiuntivo (bias); per $d$ grande il modello ha accesso a tutto il segnale.

- $m$ fisso (es. $m = 100$), $d$ da $1$ a $D$, griglia più fitta intorno a $d = m$.

- Media (o mediana) su più ripetizioni con seed diversi, con banda di variabilità.

## Dataset reale con feature casuali

- Un dataset di regressione con poche feature (es. California Housing, 8 feature, presente anche nel repo del prof).

- Per far crescere $d$ oltre $m$: feature casuali $\phi(x) = \max(0, Wx)$ (ReLU) con $W$ gaussiana fissata, oppure random Fourier features come nel paper. Il modello resta **lineare nei parametri** $w$. Il corso chiama questa idea *feature expansion*: aggiungere nuove feature costruite dalle originali per ridurre il bias dei predittori lineari (slide “11. kernels”, p. 1).

- Sottocampioniamo $m$ esempi di training (es. $m = 200$) e teniamo un test set separato.

## Regole contro il data leakage

- Split train/test *prima* di qualunque pre-processing.

- Media e deviazione standard per standardizzare calcolate **solo** sul training set e applicate al test set.

- Le matrici casuali $W$ non dipendono dai dati, quindi non creano leakage, ma vanno generate con seed fissato.

- Iperparametri (es. $\alpha$) scelti solo con dati di training.

## Domande d’orale probabili, con risposta breve

1.  **Cos’è il double descent?** Il test error, in funzione della complessità, scende, risale fino a un picco alla soglia di interpolazione ($d=m$) e poi scende di nuovo nel regime sovra-parametrizzato.

2.  **Perché il picco proprio a $d = m$?** Lì $X$ è quasi quadrata, $X^\top X$ ha un autovalore quasi nullo, la soluzione divide il rumore per quell’autovalore e $\left\lVert w \right\rVert$ esplode (slide 9, p. 5 e Sezione 5).

3.  **Perché il training error è $0$ per $d \ge m$?** Con $d \ge m$ e righe indipendenti, $Xw = y$ ha almeno una soluzione esatta: il modello interpola.

4.  **Che soluzione usi per $d > m$ e perché?** La minima norma $X^{+}y$: è unica, interpola, ed è il limite della ridge per $\alpha \to 0$ e il punto a cui converge la GD partita da $0$.

5.  **Come hai calcolato $X^{+}$?** Con `np.linalg.pinv`, basata sulla SVD. È algebra lineare, non un fit.

6.  **Cosa fa la ridge al picco?** Sostituisce $\lambda_i$ con $\lambda_i + \alpha$ al denominatore: niente amplificazione del rumore, picco attenuato o eliminato, al costo di un po’ di bias.

7.  **Come hai scelto $\alpha$?** Valori fissi a confronto, e/o validazione solo sul training set. Mai sul test.

8.  **Come fai variare $d$ sul dataset reale?** Con feature casuali; il modello resta lineare nei parametri.

9.  **Come eviti il data leakage?** Split prima di tutto, statistiche di standardizzazione solo dal training, iperparametri solo dal training.

10. **Perché il double descent contraddice la teoria classica?** La teoria classica prevede che oltre un certo punto aumentare la complessità peggiori sempre la generalizzazione (curva a U). Nel regime $d > m$ succede il contrario.

11. **Punti deboli del tuo lavoro?** Modelli solo lineari; il dataset reale richiede feature casuali, quindi il risultato dipende da come le costruisci; poche ripetizioni per vincoli di tempo; scelta di $m$ arbitraria.

## Soluzioni delle domande di verifica

*Prova a rispondere prima da solo, poi confronta.*

#### Domanda 1.

Perché per $d > m$ la formula $(X^\top X)^{-1}X^\top y$ non si può usare, e cosa calcola `pinv(X) @ y`?

*Risposta.* $X^\top X$ è $d\times d$ ma ha rango al massimo $m < d$, quindi è singolare e non invertibile (equivalentemente, $m$ vettori non generano $\mathbb{R}^d$, slide 9 p. 4). Il sistema $Xw = y$ ha infinite soluzioni che interpolano. `pinv(X) @ y` restituisce quella di norma euclidea minima, $X^\top(XX^\top)^{-1}y$: è unica, sta nello spazio delle righe di $X$, e ogni altra soluzione ha norma maggiore (Pitagora). Per $d \le m$ coincide con la formula delle slide.

#### Domanda 2.

Perché la ridge non ha il picco a $d = m$?

*Risposta.* Nella soluzione di minima norma le componenti di $y$ vengono divise per gli autovalori $\lambda_i$ di $X^\top X$; a $d \approx m$ il più piccolo è quasi $0$ e il rumore esplode. La ridge inverte $\alpha I + X^\top X$, i cui autovalori sono $\lambda_i + \alpha \ge \alpha > 0$ (slide 9 p. 5): il denominatore non scende mai sotto $\alpha$, quindi $\left\lVert w \right\rVert$ e il test error restano limitati. Il prezzo è il bias: con $\alpha$ troppo grande $w \to 0$ e il modello sotto-adatta.

## Fonti

- N. Cesa-Bianchi, slide “Introduction” (1. intro), p. 3–5; “Statistical learning” (4. statRisk), p. 1–6; “Hyperparameter tuning and risk estimates” (6. crossVal), p. 2–4; “Kernels” (11. kernels), p. 1.

- N. Cesa-Bianchi, slide “Linear predictors” (9. linear), versione del 10/02/2026, p. 4–5.

- N. Cesa-Bianchi, slide “Online Gradient Descent” (10. online), versione del 24/02/2026.

- M. Belkin, D. Hsu, S. Ma, S. Mandal, “Reconciling modern machine learning practice and the classical bias-variance trade-off”, PNAS 2019 (arXiv:1812.11118). Versione arXiv v2 (10/09/2019); numeri di pagina riferiti a quel PDF.

- M. Belkin, D. Hsu, J. Xu, “Two models of double descent for weak features”, 2019 (arXiv:1903.07571). *Esterna.*

- T. Hastie, A. Montanari, S. Rosset, R. Tibshirani, “Surprises in high-dimensional ridgeless least squares interpolation”, Annals of Statistics 2022. *Esterna.*

- P. Nakkiran, P. Venkat, S. Kakade, T. Ma, “Optimal regularization can mitigate double descent”, ICLR 2021. *Esterna.*

- Assignment “ML projects 25/26”, Assignment 4 (p. 8–9) e FAQ (p. 16).
