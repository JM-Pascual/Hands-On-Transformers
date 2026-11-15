# Hands-On Transformers — Clase teórica: Attention Is All You Need

**Guion del docente · borrador para aprobación**
Duración estimada: unos 85 minutos, 31 diapositivas. El texto a decir va en párrafos; las *[indicaciones]* van en cursiva.

---

## Diapositiva 1 — Portada (≈1 min)

Bienvenidos. En las próximas clases vamos a trabajar sobre un único paper: "Attention Is All You Need", de Vaswani y colegas de Google, publicado en NeurIPS en 2017.

Vale la pena dedicarle tiempo a un solo paper porque casi todo lo que escuchan hoy sobre IA se apoya en la arquitectura que introdujo: el Transformer. Eso incluye GPT, Claude, BERT, los sistemas de traducción e incluso muchos modelos de visión.

*[Señalar las dos grillas.]* Estas dos grillas son un adelanto de hacia dónde vamos. Cada fila es una palabra que pregunta "¿qué otras palabras me importan?", y los cuadrados más oscuros significan "mucho". La azul es la self-attention: la oración en español mirándose a sí misma. La naranja es la cross-attention: cada palabra de la traducción en inglés mirando al español. Fíjense que casi sigue la diagonal: The con El, cat con gato, bed con cama. Al final de la clase van a saber de dónde sale cada una. Hoy arrancamos por el contexto: por qué hacía falta este paper.

## Diapositiva 2 — Hoja de ruta (≈1 min)

Hoy recorremos el paper de punta a punta, en cuatro tramos.

- Primero, contexto: cómo se traducía con RNN, qué agregó la atención de Bahdanau y por qué el paper apuesta a usar solo atención.
- Después, la notación: los tamaños y el elenco de Q, K y V que vamos a usar en todo el recorrido.
- *[Señalar la parada 03.]* El grueso de la clase: el Transformer pieza por pieza. Arrancamos con el diagrama completo y después hacemos zoom en siete bloques, siempre en el mismo orden, de la entrada a la salida. Este número de 1 a 7 los va a acompañar en el mini-mapa de arriba a la derecha de cada slide.
- Y cerramos con todo junto: traducimos "El gato duerme" caja por caja y palabra por palabra.

*[Señalar la línea de abajo.]* Doy por sentado que manejan multiplicación de matrices, softmax, descenso por gradiente y backpropagation. Backpropagation va a aparecer sobre todo en las conexiones residuales. Cómo se entrena el modelo y qué resultados obtuvo queda para después.

## Diapositiva 3 — Seq2seq: toda la oración en un solo vector (≈3 min)

Para entender por qué el Transformer fue tan importante, tenemos que saber qué reemplazó. Arrancamos en 2014, con dos papers casi simultáneos: Sutskever et al. y Cho et al. Los dos proponen el esquema encoder–decoder con redes recurrentes, conocido como seq2seq.

*[Señalar el diagrama, de izquierda a derecha.]* Queremos traducir "El gato duerme" a "The cat sleeps". El encoder es una RNN que lee la oración de a un token por vez. En cada paso combina el estado anterior con el token actual.

*[Señalar la primera fórmula.]* Eso es h_t = f(h_{t−1}, x_t), donde x_t es el embedding del token en la posición t. Los mismos pesos se reutilizan en cada paso.

*[Señalar la caja naranja.]* Cuando el encoder termina, su último estado, h_3, se convierte en el vector c. Ese vector es todo lo que el decoder va a recibir de la oración original. El decoder es otra RNN que genera la traducción palabra por palabra: "The", después "cat", después "sleeps".

*[Señalar la tercera fórmula.]* Cada estado del decoder, s_i, se calcula con su estado anterior, con c y con y_{i−1}, la palabra que acaba de generar. Ojo: y no es un estado oculto, es el token de salida, que entra como embedding. Al generar "cat", el decoder recibe "The". De la oración original, lo único que ve es c.

*[Señalar la columna "El problema".]* Y acá está el problema central: un cuello de botella. Una oración de 3 palabras o de 50 tiene que entrar en un vector del mismo tamaño fijo. En la práctica, la calidad de la traducción caía a medida que las oraciones se hacían más largas.

*[Preguntar a la clase:]* ¿Qué harían ustedes para que el decoder no dependa de un único vector? *(Dejar que propongan; la respuesta natural es "que pueda mirar todos los estados del encoder", que es la siguiente diapositiva.)*

## Diapositiva 4 — La "energía": puntuar cada palabra de entrada (≈3,5 min)

La respuesta llegó también en 2014, con Bahdanau, Cho y Bengio: la atención.

*[Señalar la fila de abajo del diagrama.]* Primer cambio: el encoder ya no tira sus estados intermedios. Es una RNN bidireccional, que lee la oración en ambos sentidos, y guarda un estado por palabra: h_1, h_2, h_3. Cada uno representa a su palabra en el contexto de toda la oración.

*[Señalar s_2 y las tres líneas.]* Segundo cambio: en cada paso, el decoder decide a qué partes de la entrada mirar. Supongamos que estamos en el paso 2, generando "cat". Para decidir, primero tiene que puntuar cada palabra de entrada. Ese puntaje es el centro de esta slide.

*[Señalar el diagrama de la red, a la derecha.]* El puntaje lo calcula una red chiquita, que el paper llama "alignment model", a. Es un MLP con una sola capa oculta:
- Entra el estado previo del decoder, s_{i−1}, y un único estado del encoder, h_j.
- Cada uno se proyecta con su propia matriz, W_a y U_a; se suman y pasan por una tanh. Son 1000 unidades ocultas.
- v_a proyecta esa capa oculta a un solo número, sin activación. Ese número es e_ij, y el paper lo llama "energía": un real cualquiera, sin normalizar, lo que hoy llamaríamos un logit.

*[Señalar la fórmula.]* Todo junto: e_ij = v_aᵀ tanh(W_a s_{i−1} + U_a h_j). Como las dos proyecciones se suman, a esto se lo llama atención aditiva.

*[Señalar la tabla.]* Lo importante: la red recibe un par y devuelve un número. Para generar "cat" la llamamos tres veces, una por palabra de entrada, siempre con los mismos pesos: e_21, e_22 y e_23. Así obtenemos un vector de energías del largo de la oración de entrada. Con estos valores ilustrativos, "gato" se lleva la energía más alta.

*[Preguntar a la clase:]* ¿Por qué no hacer una red que devuelva las tres energías de una vez? *(Respuesta: porque su salida dependería del largo de la oración. Puntuando par por par con pesos compartidos, el mismo modelo sirve para 3 palabras o para 50.)*

## Diapositiva 5 — Atención: un contexto distinto por palabra (≈2,5 min)

*[Señalar las fórmulas de la derecha, de arriba hacia abajo.]* Ya tenemos las energías. Ahora:
- La primera fila es lo que acabamos de ver: la energía e_ij.
- Esas energías pasan por una softmax sobre j, así los pesos α_ij son positivos y suman uno. Con las energías de la slide anterior, 0, 2,08 y 0, la softmax da, redondeando, 0,1, 0,8 y 0,1 (e^2,08 ≈ 8, así que 8/10 y 1/10).
- Por último se arma un contexto propio para ese paso: c_i, el promedio de los h_j ponderado por esos pesos.

*[Señalar la línea gruesa.]* Acá α_22 vale 0,8: al generar "cat", el decoder mira sobre todo a "gato". *[Señalar el último párrafo.]* El paper interpreta α_ij como la probabilidad de que la palabra generada y_i esté alineada con la palabra de entrada x_j, o traducida a partir de ella. Por eso decimos que los pesos se pueden leer como un alineamiento entre las dos oraciones, y es algo que el modelo aprende solo, sin que nadie se lo indique.

*[Señalar la frase en naranja y azul.]* La idea clave: pasamos de un único c para toda la traducción a un c_i distinto por cada palabra generada. Guarden esta fórmula, c_i = Σ α_ij · h_j, porque es exactamente lo que el Transformer va a generalizar.

## Diapositiva 6 — Lo que resolvió la atención, y lo que no (≈2,5 min)

Hagamos el balance.

*[Columna izquierda.]* La atención resolvió mucho:
- Eliminó el cuello de botella: ya no hay que comprimir todo en un c fijo.
- Le dio al decoder acceso directo a cualquier palabra de la entrada, esté donde esté.
- Y nos regaló un alineamiento aprendido que además podemos inspeccionar.

*[Columna derecha.]* Pero varias cosas seguían igual, porque el modelo seguía siendo una RNN:
- **Cómputo secuencial.** El encoder y el decoder siguen avanzando token por token: O(n) pasos que no se pueden paralelizar dentro de una secuencia, ni siquiera durante el entrenamiento, cuando ya conocemos toda la oración. Las GPU, que son buenas haciendo miles de cosas a la vez, quedan esperando.
- **La atención solo conectaba encoder y decoder.** Las relaciones entre palabras de una misma oración, por ejemplo entre "gato" y "duerme", seguían pasando por la cadena recurrente.
- **Dependencias lejanas.** Dentro de una secuencia, la información y el gradiente tienen que atravesar O(n) pasos. Por la regla de la cadena, en backpropagation through time eso es un producto de un jacobiano por paso, que tiende a cero o explota. Las LSTM lo atenúan, pero no lo eliminan.
- **Memoria.** Como no se puede paralelizar dentro de una secuencia, la única forma de paralelizar es procesar muchas secuencias en un batch. Y con secuencias largas, la memoria limita cuántas entran.

*[Pausa en la línea de abajo.]* De acá sale la idea del paper: si la atención funciona tan bien entre encoder y decoder, llevémosla también dentro de cada secuencia, y eliminemos la recurrencia por completo.

## Diapositiva 7 — La apuesta: solo atención (≈2,5 min)

Esta tabla está adaptada de la Tabla 1 del paper. Compara tres tipos de capa según tres medidas. n es la longitud de la secuencia y d es el tamaño del vector de cada token.

*[Señalar la columna "Pasos secuenciales".]* En una capa recurrente es O(n): es el cómputo secuencial que acabamos de ver. En auto-atención es O(1): todas las posiciones se calculan a la vez.

*[Señalar la columna "Camino más largo".]* En una capa recurrente, la señal entre dos tokens recorre O(n) pasos: son las dependencias lejanas. Con auto-atención, cada token mira directamente a todos los demás: un solo paso, a cualquier distancia. Caminos más cortos le facilitan el trabajo al gradiente.

*[Señalar la columna "Costo por capa".]* Nada es gratis. La auto-atención compara cada token con todos los demás, así que su costo crece con n². Pero una capa recurrente cuesta n · d², así que la auto-atención en realidad es más barata siempre que n sea menor que d. En el paper d es 512, y una oración típica tiene decenas de tokens: bastante menos. *(Comentario para más adelante: este término n² es justamente la razón por la que los contextos muy largos son caros en los modelos de lenguaje actuales.)*

## Diapositiva 8 — Notación I: los tamaños que vamos a seguir (≈1,5 min)

Antes de meternos de lleno, pongámonos de acuerdo en la notación. La mayor parte de la confusión con los Transformers viene de perderle el rastro a las dimensiones de las matrices, así que vamos a ponerle nombre a cada tamaño.

*[Señalar la franja oscura.]* Primero, una convención: los tokens son filas. Una oración de n tokens es una matriz X con n filas y d_model columnas, y los pesos multiplican por la derecha: X por W. Así lo escribe el paper, y así funciona PyTorch.

*[Recorrer las tarjetas.]*
- n es la cantidad de tokens. Cambia de una oración a otra.
- d_model es 512. Todos los vectores de token tienen este ancho, tanto al entrar como al salir de cada bloque.
- h es 8, la cantidad de cabezas de atención.
- d_k y d_v son 64 cada uno, que es simplemente 512 dividido 8.
- d_ff es 2048, el ancho oculto de la red feed-forward.
- N es 6, la cantidad de capas idénticas apiladas en el encoder, y otras tantas en el decoder.

Estos son los valores del modelo base del paper, que tiene unos 65 millones de parámetros.

## Diapositiva 9 — Notación II: el elenco (≈2 min)

Por último, los personajes principales.

*[Columna izquierda.]* Cada token se transforma en tres vectores mediante tres matrices aprendidas:
- La consulta (query), Q, es lo que el token está buscando.
- La clave (key), K, es aquello por lo que puede ser encontrado.
- El valor (value), V, es el contenido que efectivamente transmite.

Una analogía aproximada es buscar en una biblioteca. Su consulta se compara con las claves, las etiquetas del catálogo, y lo que se llevan a casa es el valor: el libro. Un detalle para más adelante: en la atención encoder–decoder, las consultas vienen del decoder y las claves y los valores vienen del encoder.

*[Columna derecha: leerlas, no explicarlas todavía.]* Estas son las fórmulas que van a ver una y otra vez:
- La fórmula de atención en sí. Fíjense que QKᵀ es de n por n, un puntaje por cada par de tokens: es la grilla de la portada.
- La atención multi-cabeza, que ejecuta h atenciones en paralelo.
- La red feed-forward, un MLP pequeño aplicado a cada token por separado.
- La codificación posicional, que le indica al modelo dónde está cada token, ya que sin recurrencia no tiene noción de orden.
- La conexión residual más normalización que envuelve a cada bloque.

No se preocupen por entenderlas todavía. El objetivo de hoy es reconocer los símbolos. La próxima clase arrancamos con la primera fórmula y construimos la atención desde cero.


## Diapositiva 10 — El Transformer completo (≈3 min)

Ahora sí: este es el Transformer entero. No lo vamos a entender todo hoy; la idea es tener el mapa antes de recorrerlo.

*[Señalar la columna izquierda.]* A la izquierda está el encoder. Recibe "El gato duerme", convierte cada palabra en un embedding y le suma la posición. Después pasa por un bloque que tiene dos piezas: multi-head attention y una red feed-forward, cada una envuelta en un "Add & Norm". Ese bloque se apila 6 veces.

*[Señalar las tres flechitas que entran a cada atención.]* Fíjense que a la atención del encoder le entran tres flechas desde el mismo lugar: son Q, K y V, y las tres salen de la misma oración. Eso es self-attention: lo que le faltaba al modelo de Bahdanau, atención dentro de la propia secuencia.

*[Señalar la columna derecha.]* A la derecha está el decoder. Recibe lo que ya tradujo, desplazado una posición: "<s> The cat", para predecir "The cat sleeps". Tiene tres piezas en vez de dos:
- Una self-attention enmascarada, para que cada posición solo mire las palabras anteriores.
- Una atención que mira al encoder. *[Señalar la línea azul.]* Las claves y los valores, K y V, vienen del encoder; la consulta, Q, viene del decoder. Es exactamente la atención de Bahdanau, ahora sin RNN.
- La misma feed-forward que el encoder.

*[Señalar arriba.]* Al final, una capa lineal y una softmax dan la probabilidad de cada palabra del vocabulario como próxima palabra.

*[Señalar las líneas grises.]* Y las líneas grises son las conexiones residuales: cada subcapa suma su entrada a su salida antes de normalizar. Son las que permiten apilar 6 bloques sin que el gradiente se pierda.

*[Avanzar una vez: se despliega la pila de capas.]* Y ese "×6" no es decorativo: lo que dibujamos como un bloque son en realidad 6 capas idénticas apiladas, en el encoder y en el decoder. Cada capa recibe la salida de la anterior, con el mismo tamaño, 512 por token, y por eso se pueden apilar. Cada una tiene sus propios pesos.

*[Señalar la lista de la derecha.]* Los números del diagrama son el orden en que lo vamos a recorrer, de abajo hacia arriba, siguiendo el camino de los datos.


## Diapositiva 11 — De palabras a vectores (≈2,5 min)

*[Señalar el mapa chico de arriba a la derecha.]* A partir de ahora, este mapa les va a mostrar en qué parte del Transformer estamos. Empezamos por abajo de todo: la entrada.

*[Recorrer el diagrama de izquierda a derecha.]* La red no lee palabras, lee números. Primero cada token se traduce a su índice en el vocabulario. Ese índice elige una fila de una tabla, la matriz de embeddings E, que tiene una fila por token del vocabulario y 512 columnas. Las tres filas elegidas, apiladas, forman X: una matriz de 3 por 512. Es la X de la notación que vimos antes.

*[Señalar la fórmula.]* Formalmente, el embedding del token t es la fila de E que le corresponde, multiplicada por la raíz de d_model, que es un detalle del paper.

*[Señalar los bullets.]*
- Buscar una fila es lo mismo que multiplicar el vector one-hot del token por E. O sea, es una capa lineal sin bias.
- Por eso E es una matriz de pesos como cualquier otra: arranca al azar y se aprende con backpropagation.
- En el paper no se usan palabras enteras sino subpalabras. El vocabulario, de unos 37 000 tokens, se genera con BPE: se arranca con los caracteres y se fusiona, una y otra vez, el par de símbolos más frecuente del corpus de entrenamiento. Se aprende una sola vez sobre el texto de los dos idiomas juntos, así que el vocabulario es compartido.
- Y la misma matriz E se reutiliza en tres lugares: el embedding de entrada, el de salida y la capa Linear del final.

## Diapositiva 12 — BPE: cómo se arma el vocabulario (≈3 min)

¿De dónde salen esos 37 000 tokens? Veámoslo con un corpus de juguete. *[Señalar la tarjeta de la izquierda.]* Tenemos cuatro palabras con su frecuencia: "gato" aparece 5 veces, "gatos" 2, "pato" 4 y "gata" 3.

*[Señalar el vocabulario.]* Arrancamos con el vocabulario más chico posible: los caracteres. Seis tokens.

*[Señalar "Próxima fusión".]* La regla es una sola: contar todos los pares de símbolos vecinos, pesados por la frecuencia de cada palabra, y fusionar el más frecuente. Ahora gana "a + t", que aparece 14 veces.

*[Avanzar.]* Fusión 1: "at" pasa a ser un token nuevo y aparece en todas las palabras. Ahora el par más frecuente es "at + o", 11 veces.

*[Avanzar.]* Fusión 2: "ato". *[Avanzar.]* Fusión 3: "gato" ya es un único token. *[Avanzar.]* Fusión 4: "pato".

*[Señalar el resultado.]* Paramos acá porque elegimos hacer 4 fusiones: la cantidad de fusiones es el único hiperparámetro del algoritmo, y define el tamaño del vocabulario. En el paper hicieron las suficientes para llegar a unos 37 000 tokens.

Miren lo interesante: "gatos" quedó como "gato" + "s", y "gata" como "g" + "at" + "a". Las palabras frecuentes terminan siendo un token entero; las raras se arman con pedazos. Así, mientras sus caracteres estén en el vocabulario base, cualquier palabra se puede representar, aunque nunca haya aparecido entera en el entrenamiento.

Detalle importante: esto se calcula una sola vez, sobre el corpus de entrenamiento de los dos idiomas juntos. Cuando llega una oración nueva, no se vuelve a entrenar BPE: se aplican las mismas fusiones, en el mismo orden.

## Diapositiva 13 — Sin recurrencia, el orden se pierde (≈2 min)

Ahora un problema que el Transformer se crea a sí mismo al sacar la recurrencia.

*[Señalar la tarjeta RNN.]* En una RNN, el orden viene gratis. Lee de a un token: primero "El", después "gato", después "duerme", y cada estado h_t se construye a partir del anterior. La posición de cada palabra es simplemente el paso t en el que se leyó. Nadie tuvo que decírselo.

*[Señalar la tarjeta Transformer.]* El Transformer, en cambio, procesa todos los tokens a la vez, y en la atención cada uno mira a todos los demás. Eso es lo que lo hace rápido, pero tiene un costo: no hay un "primero" ni un "después". La atención compara contenidos, no lugares. Para ella, "El gato duerme" y "duerme gato El" son el mismo conjunto de vectores.

*[Preguntar a la clase:]* ¿Por qué esto es grave? *(Ejemplo: "El perro muerde al hombre" y "El hombre muerde al perro" tienen las mismas palabras y significan cosas opuestas.)*

*[Señalar el recuadro negro.]* Entonces, si la arquitectura no trae la posición, hay que ponerla en los datos. A cada token le sumamos un vector que dice dónde está: x_t + PE_t. Cómo se construye ese vector es la próxima slide.

## Diapositiva 14 — Codificación posicional sinusoidal (≈3 min)

*[Señalar las fórmulas.]* El paper construye PE con senos y cosenos. En las dimensiones pares usa seno, en las impares coseno, y cada par de dimensiones tiene una frecuencia distinta.

*[Señalar el mapa de calor.]* Así se ve: cada fila es el vector de una posición, de 0 arriba a 49 abajo, y cada columna es una dimensión. A la izquierda las ondas oscilan muy rápido; a la derecha, muy lento. Cada posición termina con una combinación única de valores, como un código de barras.

*[Señalar los bullets.]*
- El vector de posición se suma al embedding, no se concatena. Por eso tiene que tener el mismo tamaño, 512.
- Una propiedad linda: PE de pos+k se puede escribir como una función lineal de PE de pos. Eso le facilita al modelo atender por distancia relativa ("la palabra de dos lugares atrás").
- Estos vectores no se entrenan. Los autores probaron con posiciones aprendidas y les dio casi lo mismo; eligieron senos porque podrían generalizar a oraciones más largas que las del entrenamiento.

Con esto, la entrada está lista: X + PE, una matriz de n por 512 que ya sabe qué palabra es cada token y dónde está. Lo próximo es qué hace la atención con esa matriz.


## Diapositiva 15 — Self-attention: cada palabra le pregunta a todas (≈3 min)

*[Señalar el mini-mapa.]* Entramos al corazón del modelo: la atención del encoder.

*[Señalar la tarjeta de Bahdanau.]* Recuerden cómo puntuaba Bahdanau: una red chica, con una capa oculta y tanh, que se evalúa para cada par: el estado previo del decoder contra cada estado del encoder. Y como s_{i−1} sale de la RNN, eso se hace un paso del decoder a la vez.

*[Señalar la parada 03.]* El Transformer la reemplaza por algo mucho más simple: un producto escalar. Multiplicar coordenada a coordenada y sumar. Es grande cuando los dos vectores apuntan en la misma dirección. Esta es la nueva "energía".

*[Señalar la columna derecha.]* ¿Entre qué vectores? Cada token x, de 512, se proyecta con tres matrices aprendidas a tres vectores de 64:
- La consulta, q: qué está buscando el token.
- La clave, k: por qué lo pueden encontrar.
- El valor, v: qué información entrega si lo eligen.

*[Señalar el recuadro azul.]* Y lo importante: en self-attention, las consultas, las claves y los valores salen todos de la misma oración. Cada palabra le pregunta a todas las demás, incluida ella misma. Eso es lo que le faltaba a Bahdanau: atención dentro de la propia secuencia.

## Diapositiva 16 — Scaled dot-product, paso a paso (≈4 min)

*[Señalar la fórmula.]* Esta es la fórmula central del paper. Vamos a calcularla en cuatro pasos con "El gato duerme".

*[Paso 1.]* Multiplicamos Q por K traspuesta. Cada celda es el producto escalar entre la consulta de una palabra y la clave de otra: la energía. Las filas son quién pregunta; las columnas, a quién. Son las mismas energías de Bahdanau, pero calculadas para todos los pares de una sola vez.

*[Avanzar. Paso 2.]* Dividimos todo por la raíz de d_k, que es 8. El orden no cambia, pero se achican las diferencias. Por qué hace falta lo vemos en la próxima slide.

*[Avanzar. Paso 3.]* Softmax, y ojo: se aplica fila por fila. Cada fila es la atención de una palabra particular sobre todas las demás, y pasa a ser una distribución de probabilidad, positiva y que suma 1. Leer la celda (i, j) es leer cuánto atiende la palabra i a la palabra j. Son los α de Bahdanau. "gato" se mira sobre todo a sí mismo, 0,77, y después a "duerme".

*[Avanzar. Paso 4.]* Por último, multiplicamos por V: cada palabra sale como un promedio de los valores de todas, pesado por esos α. La salida tiene la misma forma que V, 3 por 64: un vector nuevo por token, que ahora incorpora su contexto. Es exactamente el c_i de Bahdanau, para cada palabra a la vez.

## Diapositiva 17 — ¿Por qué dividir por √d_k? (≈3 min)

Volvamos al paso 2: ¿por qué dividir por la raíz de d_k? Ojo, d_k no es el tamaño del embedding: es el tamaño de cada consulta y cada clave dentro de una cabeza, 64. La historia tiene tres pasos.

*[Señalar la tarjeta 1.]* Uno: las energías crecen con la dimensión. El producto escalar es una suma de d_k términos. Si cada término es un número chico que a veces da positivo y a veces negativo, la suma no se cancela del todo: su tamaño típico crece como la raíz de d_k. Con 2 dimensiones, alrededor de ±1,4; con 64, alrededor de ±8.

*[Señalar la tarjeta 2.]* Dos: con números grandes, la softmax se satura. Miren: softmax de 2, 1, 0 reparte la atención. Softmax de 16, 8, 0, que son las mismas energías multiplicadas por 8, le da todo a una sola palabra. Un one-hot.

*[Señalar la tarjeta 3.]* Tres: un one-hot no deja aprender. Si la softmax está saturada, mover un poco las energías no cambia la salida: su derivada es casi cero. En backpropagation, el gradiente que tiene que llegar a W_Q y W_K se multiplica por esa derivada, y no llega nada. Es lo mismo que una sigmoide saturada, y el mismo problema de gradientes que se apagan que vimos en las RNN.

*[Señalar el recuadro de la solución.]* La solución es dividir por raíz de d_k, que es 8: compensa exactamente el crecimiento del paso uno. Las energías vuelven a un tamaño cercano a 1, la softmax reparte y el gradiente pasa.

*(Si preguntan por qué la raíz y no d_k: lo que crece con d_k es la varianza de la suma; el tamaño típico, el desvío, crece con su raíz.)*

## Diapositiva 18 — Todo es multiplicación de matrices (≈2,5 min)

*[Señalar el diagrama, de izquierda a derecha.]* Miren las formas. Entra X, de n por 512. Tres multiplicaciones dan Q, K y V, de n por 64. Q por K traspuesta da S, de n por n. La softmax no cambia la forma. Y A por V da Z, de n por 64.

*[Señalar las tres tarjetas.]* Esto tiene tres consecuencias, y las tres nos llevan de vuelta a la tabla de "La apuesta":
- Toda la oración se procesa de una vez, sin un bucle sobre t. Son productos de matrices, que es justo lo que una GPU hace mejor. Se acabó el cómputo secuencial.
- Cada fila de A conecta un token con todos los demás, a cualquier distancia. Las dependencias lejanas están a un solo salto.
- El precio es la matriz de n por n. Con 3 tokens son 9 energías; con 1 000 tokens, un millón, y por cada cabeza. Es el costo cuadrático de la tabla, y la razón por la que los contextos muy largos son caros.

## Diapositiva 19 — Multi-head: 8 atenciones en paralelo (≈3 min)

Una sola atención produce un único patrón de "quién mira a quién". Pero en una oración hay muchas relaciones a la vez.

*[Señalar el diagrama.]* Entonces el paper corre 8 atenciones en paralelo, cada una con sus propias matrices W_Q, W_K y W_V, que proyectan de 512 a 64. Cada cabeza produce una salida de n por 64. Se concatenan, 8 por 64 da 512, y una última matriz, W_O, mezcla lo que encontraron y devuelve el resultado a d_model.

*[Señalar los mapas de calor.]* Como cada cabeza tiene sus propios pesos, puede especializarse. Estos patrones son ilustrativos: una cabeza podría mirar a la palabra anterior, otra conectar sujeto con verbo. En el paper muestran ejemplos reales de cabezas que aprenden relaciones sintácticas.

*[Señalar los bullets.]* Y no sale más caro: 8 cabezas de 64 cuestan lo mismo que una sola atención de 512. Los autores probaron con una sola cabeza y les dio 0,9 BLEU menos.

*[Señalar el último bullet.]* Ojo con no confundir dos cosas. Las 8 cabezas van en paralelo, una al lado de la otra. Pero toda la multi-head, entera, es una sola subcapa: su salida, n por 512, es la entrada de la subcapa siguiente, la feed-forward. Y esa capa completa se repite 6 veces, una arriba de la otra. Paralelo a lo ancho, en serie a lo alto. Vamos a volver sobre esto en la próxima slide.

Con esto cerramos la atención. Lo que sale de acá todavía tiene que pasar por el Add & Norm, que es lo próximo.


## Diapositiva 20 — Conexiones residuales: sumar, no reemplazar (≈3 min)

*[Señalar el mini-mapa.]* Después de cada subcapa, en el encoder y en el decoder, aparece este bloque "Add & Norm". Son dos ideas: "Add" es la conexión residual, y "Norm" la normalización.

*[Señalar el diagrama, de abajo hacia arriba.]* Empecemos por "Add". Entra x, el vector de cada token. Va a la subcapa, por ejemplo la atención, que calcula F(x). Pero x también toma un atajo, la línea azul, y al final se suman: la salida es x + F(x).

*[Señalar el primer bullet.]* La intuición: la subcapa no reemplaza al vector, le suma una corrección. "Esto es lo que tenías, más lo que aprendí mirando a las otras palabras." Si la subcapa no tiene nada útil para aportar, puede devolver casi cero y x pasa intacto.

*[Señalar la cadena.]* Ahora acuérdense de lo que dijimos con multi-head: las subcapas van en cadena. En el encoder son 6 capas por 2 subcapas, 12 subcapas una detrás de la otra, y la salida de cada una es la entrada de la siguiente. En el decoder son 18.

*[Señalar el bullet "Sin atajo".]* Y acá está la razón de fondo, que es el entrenamiento. Backpropagation arranca en la pérdida, arriba de todo, y baja por esa cadena hasta las primeras subcapas. En cada subcapa que atraviesa, el gradiente se multiplica por su derivada. Pongamos que cada derivada vale 0,5: después de 12 subcapas queda 0,5 a la 12, unas dos diezmilésimas. A las primeras capas no les llega casi nada, sus pesos no se mueven, no aprenden. Es el mismo problema de la RNN.

*[Señalar el bullet "Con atajo".]* Con el atajo, la subcapa calcula x + F(x), y su derivada es 1 + la derivada de F. Con el mismo 0,5, cada factor vale 1,5 en vez de 0,5: nunca se acerca a cero. El "+1" viene de la x que pasó directo. Por eso se pueden apilar 6 capas, o 100, y seguir entrenando. Y si los números tienden a crecer, de eso se encarga la normalización, que viene ahora.

*[Señalar el último bullet.]* Un detalle: para poder sumar x y F(x), tienen que medir lo mismo. Por eso todo el modelo usa vectores de 512: es lo que hace posible el atajo.

## Diapositiva 21 — Layer normalization (≈3 min)

*[Señalar el gráfico "Antes".]* Ahora "Norm". Acá vemos 8 de los 512 números del vector de un token. Pueden tener cualquier escala: en este ejemplo, todos positivos, con media 5.

*[Señalar el gráfico "Después".]* Layer normalization hace dos cosas con ese vector: lo centra, restándole su media, y lo escala, dividiéndolo por su desvío. Queda con media 0 y desvío 1. *[Señalar la fórmula.]* Después lo multiplica por γ y le suma β, dos vectores que se aprenden: si a la red le conviene otra escala, la puede recuperar.

*[Señalar los bullets.]*
- Lo importante: cuando digo "el vector de un token", en la práctica es una fila. Lo que entra a Add & Norm es una matriz de n por 512, y cada fila es la información de un token. Layer norm opera fila por fila: la media y el desvío se calculan sobre los 512 números de esa fila, sin mirar a las otras filas, es decir a las otras palabras, ni a las otras oraciones del batch. Es la misma idea que la softmax por fila de la atención.
- ¿Para qué sirve? Con 6 capas apiladas, sin normalizar, los números podrían ir creciendo o achicándose capa tras capa. Normalizar los mantiene en un rango estable, y el entrenamiento es mucho más estable y rápido.
- Si conocen batch norm: esa normaliza cada feature a lo largo del batch. Con oraciones de largos distintos eso funciona mal. Layer norm no depende del batch.

*[Señalar el recuadro oscuro.]* Juntando las dos ideas: Add & Norm es LayerNorm de x + Sublayer(x). Primero se suma el atajo, después se normaliza. Y esto va después de cada subcapa: dos veces por capa en el encoder, tres en el decoder.

## Diapositiva 22 — Feed-forward: la misma red, fila por fila (≈3 min)

*[Señalar el mini-mapa.]* Paso 4. La segunda subcapa de cada capa del encoder, y la última del decoder, es una red feed-forward. Es la pieza más simple del Transformer: un MLP de una capa oculta, como los que ya conocen.

*[Señalar la fórmula.]* Es la ecuación 2 del paper: multiplicar por W₁ y sumar b₁, pasar por una ReLU, multiplicar por W₂ y sumar b₂.

*[Señalar la tarjeta de arriba a la derecha, "fila por fila".]* Lo importante es cómo se aplica. Volvemos a la idea de las filas: lo que entra es una matriz de n por 512, una fila por token. La FFN toma cada fila y la procesa sola, sin mirar a las otras. Y es la misma red, con los mismos pesos, para todas las filas. Por eso el paper la llama "position-wise": se aplica posición por posición.

*[Señalar el primer bullet.]* Esto la diferencia de la atención. La atención es el único lugar donde los tokens intercambian información. La FFN no mezcla nada entre tokens: trabaja con lo que cada token ya juntó.

*[Señalar el MLP de la izquierda.]* Adentro, para una fila: entran las 512 coordenadas de la fila que salió del Add & Norm. Cada una se conecta con las 2048 neuronas de la capa oculta, cuatro veces más ancha, que aplican la ReLU. Y de ahí se vuelve a comprimir a 512 salidas. Es el MLP de siempre: cada neurona hace ReLU de la suma de sus entradas por sus pesos, más el bias. ¿Por qué volver a 512? Por lo que vimos recién: hay un atajo alrededor, y para sumar x + F(x) tienen que medir lo mismo.

*[Señalar el último bullet.]* ¿Y la ReLU para qué? Es la no linealidad. Sin ella, multiplicar por W₁ y después por W₂ es lo mismo que multiplicar por una sola matriz, W₁ por W₂: las dos capas colapsarían en una.

Un detalle: los pesos se comparten entre posiciones, pero no entre capas. Cada una de las 6 capas tiene su propia FFN.

## Diapositiva 23 — La capa del encoder, completa (≈3 min)

*[Señalar el diagrama, de abajo hacia arriba.]* Con esto ya tenemos todas las piezas del encoder. Entra X, n por 512: los embeddings más la posición. Pasa por multi-head attention, Add & Norm, feed-forward, Add & Norm. Las líneas azules son los atajos, uno por subcapa. Y sale una matriz del mismo tamaño, n por 512.

*[Señalar el "×6".]* Como entra y sale lo mismo, se puede apilar: esto se repite 6 veces, cada capa con sus propios pesos. Es la cadena de 12 subcapas de la que hablamos con las conexiones residuales.

*[Señalar las dos tarjetas.]* La forma más simple de recordar qué hace cada subcapa:
- La atención es la parte social: cada token se comunica con todos los demás y junta contexto.
- La feed-forward es la parte individual: cada token procesa por su cuenta lo que juntó.
Comunicar, procesar; comunicar, procesar; seis veces.

*[Señalar los bullets.]* Un dato que suele sorprender: la mayoría de los parámetros no está en la atención sino en la FFN. Por capa, la atención tiene unas 4 matrices de 512 por 512, alrededor de 1 millón de parámetros; la FFN, dos matrices de 512 por 2048, alrededor de 2 millones. El encoder entero suma unos 19 millones.

*[Señalar el último bullet.]* Y lo que sale de la última capa es la representación final de la oración de entrada. Esa matriz es lo que el decoder va a consultar más adelante: de ahí salen las K y las V de la cross-attention, que es el paso 6.

## Diapositiva 24 — El Decoder: Generando de a una palabra (≈3 min)

*[Señalar el mini-mapa.]* Paso 5: pasamos al decoder. Su primera subcapa es una self-attention, igual a la del encoder, con una diferencia: está enmascarada. Para entender por qué, primero hay que ver cómo trabaja el decoder.

*[Señalar la tarjeta izquierda, paso por paso.]* Al generar una traducción, el decoder produce una palabra por vez. Arranca con un token especial de inicio, <s>, y predice "The". Ese "The" vuelve a entrar: con "<s> The" predice "cat". Con "<s> The cat" predice "sleeps". Y así hasta que predice el token de fin, </s>. A esto se lo llama autorregresivo: cada salida se vuelve entrada. Generar es secuencial, n pasos uno detrás del otro. Eso no lo podemos evitar: no sabemos la palabra 3 hasta haber elegido la 2.

*[Señalar la tarjeta derecha.]* Pero al entrenar sí conocemos la traducción correcta completa. Entonces hacemos un truco: le damos al decoder toda la oración correcta de una vez, corrida un lugar a la derecha, con <s> adelante. Y le pedimos que en cada posición prediga la palabra siguiente. Desde <s>, "The"; desde "The", "cat"; desde "cat", "sleeps"; desde "sleeps", el fin. Las 4 predicciones salen juntas, en un solo forward, en paralelo. Esto se llama teacher forcing: en vez de alimentar al modelo con sus propias predicciones, que al principio son malas, le damos las correctas.

*[Señalar el recuadro oscuro.]* Pero acá aparece un problema. Si la oración entera está adentro, y la self-attention deja que cada palabra mire a todas, entonces la posición que tiene que predecir "cat" puede mirar "cat" en la entrada, que está justo al lado. Copiaría la respuesta y no aprendería nada. Y al generar, esa palabra todavía no existe. Hay que taparle el futuro.

## Diapositiva 25 — La máscara causal: no mirar el futuro (≈3 min)

*[Señalar la matriz de energías.]* Volvamos a la atención paso a paso. Estas son las energías, Q por K transpuesta dividido raíz de d_k: los mismos números que en la slide 16, ahora con las palabras del decoder. Recuerden: la fila i es la palabra i preguntando, la columna j es la palabra a la que mira.

*[Avanzar. Aparece M.]* La solución es sumar una máscara M. Tiene ceros en la diagonal y abajo, y menos infinito arriba de la diagonal. Arriba de la diagonal están justamente las columnas j mayores que i: las palabras que vienen después. El futuro.

*[Avanzar. Aparece A.]* Y después hacemos la softmax por fila, como siempre. e a la menos infinito es cero, así que esos pesos quedan en exactamente cero. Y como la softmax normaliza, cada fila sigue sumando 1, repartido solo entre lo visible. Miren las filas: <s> solo puede mirarse a sí misma, peso 1. "The" mira a <s> y a sí misma. "cat" mira a las tres, y como no había nada que tapar, su fila queda igual que en la slide 16.

*[Señalar la fórmula.]* En la fórmula es solo un término más: sumar M antes de la softmax.

*[Señalar los bullets.]*
- Cada palabra mira solo hacia atrás: ella misma y las anteriores.
- Hay que sumarlo antes de la softmax. Si pusiéramos ceros después, las filas dejarían de sumar 1.
- Lo lindo: con la máscara, entrenar todo en paralelo da exactamente lo mismo que generar de a una palabra. Cada posición ve lo mismo que vería en la generación real.
- Y esto es solo en la self-attention del decoder. El encoder lee una oración que ya existe completa, así que ve todo.

## Diapositiva 26 — ¿De dónde salen Q, K y V? (≈3 min)

*[Señalar el mini-mapa.]* Paso 6: la segunda subcapa del decoder. Antes de ver cómo funciona, veamos por qué hace falta.

*[Señalar "Sin esta etapa" en el banner.]* Hasta el paso 5, el decoder solo vio su propia traducción a medio hacer. La self-attention enmascarada mezcla palabras en inglés entre sí, pero ninguna capa del decoder leyó todavía el español. Pensemos en "El gato duerme" y "El perro duerme". Después de "<s> The", el decoder tiene exactamente la misma entrada en los dos casos: no tiene forma de saber si sigue "cat" o "dog". Sin esta etapa, el decoder sería un modelo de lenguaje en inglés: escribiría inglés fluido, pero sobre cualquier cosa.

La información que le falta ya existe: la calculó el encoder. Esta subcapa, la cross-attention, es el único lugar donde las dos mitades se conectan.

*[Señalar la columna izquierda.]* Del lado del encoder, "El gato duerme" pasa por las 6 capas que ya vimos, y sale una matriz que vamos a llamar H: n filas de 512, una por palabra de la oración original.

*[Señalar la columna derecha, de abajo hacia arriba.]* Del lado del decoder, "<s> The cat" pasa por la self-attention enmascarada del paso 5 y su Add & Norm. Lo que sale, Y, tiene m filas de 512.

*[Señalar las tres flechas que entran a la cross-attention.]* Y acá está la clave. A la cross-attention le entran tres flechas, como a cualquier atención, pero de dos lugares distintos:
- Q sale del decoder, de Y. Es la pregunta: "¿qué necesito para la próxima palabra?".
- K y V salen del encoder, de H. Es lo que hay para consultar: lo que dice la oración original. K sirve para decidir dónde mirar; V es lo que se trae.

Pregunta el que necesita la información. Como en una biblioteca: el que busca pone la consulta, y la biblioteca pone los índices y los libros.

*[Señalar la tarjeta de K y V.]* Dos detalles. La misma H, la salida de la última capa del encoder, entra a las 6 capas del decoder: cada capa puede hacerle una pregunta distinta. Y como la oración original y la traducción pueden tener largos distintos, n y m no tienen por qué ser iguales.

*[Señalar "Con esta etapa" en el banner.]* ¿Qué logramos? Cada fila del decoder entra sabiendo qué escribió en inglés, y sale sabiendo también qué dice el input en español, tal como lo vio el encoder, en la parte que le importa. Y no recibe un único vector que resume la oración, como en la RNN de la slide 4: puede consultar las n filas cada vez. En una frase: la self-attention enmascarada le dice al decoder qué escribió; la cross-attention le dice qué tiene que traducir.

*[Avanzar.]* Veamos ahora qué calcula.

## Diapositiva 27 — Cross-attention: consultar al encoder (≈3 min)

*[Señalar la matriz.]* Es la misma atención de siempre. Como Q sale del decoder y K y V del encoder, las filas de esta matriz son palabras de la traducción, y las columnas, palabras de la oración original. Cada fila es una palabra del decoder repartiendo su atención sobre la oración en español. Y como n y m pueden ser distintos, esta matriz no tiene por qué ser cuadrada.

*[Recorrer las filas.]* Fíjense qué pasa. Desde <s>, que tiene que predecir "The", la atención va sobre todo a "El". Desde "The", que tiene que predecir "cat", va a "gato". Y desde "cat", para predecir "sleeps", va a "duerme". Es la alineación entre los dos idiomas. Los números son ilustrativos, pero es el tipo de patrón que el modelo aprende.

*[Señalar la tabla.]* ¿Les suena? Es exactamente la atención de Bahdanau, la de la slide 4. La pregunta antes era el estado de la RNN del decoder; ahora es una fila del decoder proyectada con W_Q. Las claves y valores antes eran los hidden states del encoder; ahora son las filas de la salida del encoder. La energía antes la calculaba un MLP; ahora es un producto punto escalado. Y lo más importante: antes se calculaba un paso del decoder a la vez; ahora todas las filas a la vez, y con 8 cabezas. Cerramos el círculo con el que arrancamos la clase: la idea de la atención es la misma, sin la RNN.

*[Señalar los bullets.]*
- Q es "qué busco"; K y V son "qué dice la oración original".
- Acá no hay máscara: la oración de entrada ya existe completa, no hay futuro que tapar. La máscara está en la subcapa anterior, sobre la traducción.
- Y un detalle: la misma salida del encoder se usa en las 6 capas del decoder. Cada capa la consulta con sus propios pesos, buscando cosas distintas.

## Diapositiva 28 — De un vector a la próxima palabra (≈3 min)

*[Señalar el mini-mapa.]* Paso 7, el último: la salida. El decoder terminó sus 6 capas y nos devuelve una matriz Y de m×512. Pero nosotros queremos una palabra, no 512 números. ¿Cómo pasamos de uno a otro?

*[Recorrer la tira de arriba, de izquierda a derecha.]*
- Primero: de Y nos quedamos con la última fila, la de "cat". Por la máscara, es la posición que tiene que predecir la palabra que sigue.
- Esa fila de 512 se multiplica por una matriz de 512 × 37 000: una columna por cada palabra del vocabulario. Esa es la capa Linear.
- Sale un vector de 37 000 números, los logits: un puntaje por palabra.
- La softmax, la misma de siempre, los convierte en probabilidades que suman 1.
- Y elegimos la palabra: acá, "sleeps".

*[Señalar el dibujo de la red, abajo a la izquierda.]* ¿Cómo es esa Linear por dentro? Comparen con la FFN de la slide 22: acá no hay capa oculta ni ReLU. Es una sola capa. Las 512 coordenadas de la fila de "cat" se conectan con 37 000 neuronas de salida, una por cada palabra del vocabulario. Cada neurona de salida da un logit, y la softmax los convierte en probabilidades. "sleeps" se lleva 0.62; después vienen "is", "sits", y el resto de las 36 995 palabras se reparte lo que queda. Los números son ilustrativos.

*[Señalar el bullet de weight tying.]* Un detalle lindo: la matriz de la Linear no es nueva. Es E transpuesta, la misma tabla de embeddings del paso 1. Entonces cada logit es el producto punto entre la fila del decoder y el embedding de cada palabra: el modelo pregunta "¿a qué palabra se parece este vector?". Y nos ahorramos 37 000 × 512, unos 19 millones de parámetros, sobre un modelo de 65.

*[Señalar el último bullet.]* Al entrenar no usamos solo la última fila: usamos las m a la vez, porque con teacher forcing cada posición tiene su palabra correcta. La pérdida es menos el logaritmo de la probabilidad que el modelo le dio a la palabra correcta, en cada posición: cross-entropy. Cuanto más probable la correcta, menor la pérdida.

## Diapositiva 29 — Generar: elegir y repetir (≈3 min)

*[Señalar la tarjeta de la izquierda.]* Juntemos todo en el loop de generación. El encoder corre una sola vez: "El gato duerme" se convierte en H y queda fija. Después, el decoder recibe lo que generó hasta ahora, calcula la última fila, Linear y softmax, elegimos una palabra, la agregamos al final y volvemos a empezar. Hasta que el modelo genera el token de fin, </s>. Es exactamente lo que vimos en la slide 24, ahora con la última pieza.

*[Señalar el título de la tarjeta de la derecha.]* Queda una pregunta: ¿qué palabra elegimos? La más obvia es la más probable en cada paso. Eso se llama greedy.

*[Recorrer el árbol.]* Pero greedy puede equivocarse. Partimos de "The cat". "is" tiene 0.40 y "sleeps" 0.35; greedy elige "is". Después lo mejor que viene es "sleeping" con 0.50, y la secuencia termina con 0.40 × 0.50 = 0.20. Si en cambio hubiéramos ido por "sleeps", el punto final tiene 0.90, y la secuencia suma 0.35 × 0.90 = 0.315. Mejor, aunque el primer paso parecía peor.

Beam search arregla eso: en vez de quedarse con una sola secuencia, guarda las k mejores en cada paso y compara las secuencias completas. Con k = 2 encuentra "sleeps .".

*[Señalar los bullets de abajo.]*
- El paper usa beam de 4, con una penalización por largo para no favorecer frases cortas: cada palabra extra multiplica por un número menor a 1, así que sin corrección la búsqueda preferiría frases cortas.
- En la práctica se suman logaritmos en lugar de multiplicar probabilidades: el producto de muchos números chicos se vuelve diminuto.
- Y un comentario actual: los LLM de chat casi no usan beam. Muestrean, con temperatura o top-p, porque buscan respuestas variadas, no la frase más probable.

## Diapositiva 30 — Cierre: traducción paso a paso (≈5 min, 19 clics)

Ritmo rápido: una o dos frases por clic. El recuadro azul marca qué caja estamos mirando.

*[Paso 0.]* Cerramos con el diagrama de la slide 10 y el ejemplo de toda la clase: traducir "El gato duerme". Primero el encoder, una vez. Después el decoder, de a una palabra.

**Encoder**
- *[Clic 1: Embedding.]* Cada token busca su fila en E y se le suma la posición: 3×512.
- *[Clic 2: Self-attention.]* Cada palabra mira a las otras tres.
- *[Clic 3: Add & Norm.]* Se suma la entrada y se normaliza cada fila.
- *[Clic 4: Feed-forward.]* Cada fila sola por el mismo MLP.
- *[Clic 5: segundo Add & Norm.]* Otra vez: se suma la entrada de la FFN y se normaliza cada fila.
- *[Clic 6: todo el encoder, se abren las 6 copias.]* Este bloque entero se repite 6 veces. Sale H, y queda fija: miren la etiqueta arriba.

**Decoder, primera palabra, caja por caja**
- *[Clic 7: Embedding.]* Todavía no hay traducción: entra solo <s>.
- *[Clic 8: Masked self-attention.]* <s> solo puede mirarse a sí mismo.
- *[Clic 9: Add & Norm.]* Atajo y normalización, como en el encoder.
- *[Clic 10: Cross-attention, se ilumina K, V.]* Acá el decoder consulta H: la fila de <s> mira sobre todo a "El".
- *[Clic 11: Add & Norm.]* Otra vez atajo y normalización.
- *[Clic 12: Feed-forward.]* La fila pasa sola por el MLP.
- *[Clic 13: Add & Norm.]* El último de la capa.
- *[Clic 14: todo el decoder, se abren las 6 copias.]* Las tres subcapas, 6 veces. Las 6 capas consultan la misma H.
- *[Clic 15: Linear + softmax.]* Proyectamos a las 37 000 palabras: la más probable es "The". Aparece arriba, en "Traducción generada".

**Las palabras que siguen, un clic cada una** (el camino es el mismo, por eso se recuadra todo el decoder)
- *[Clic 16.]* "The" vuelve a entrar. La cross-attention va a "gato": sale "cat".
- *[Clic 17.]* Entra "cat", la cross-attention va a "duerme": sale "sleeps", el 0.62 de la slide 28.
- *[Clic 18.]* Entra "sleeps". No queda nada por traducir: sale </s> y cortamos. "The cat sleeps".

*[Clic 19: Para llevarse.]* Cuatro ideas: la atención comunica las filas; la feed-forward procesa cada fila sola; Add & Norm permite apilar 12 a 18 subcapas; la máscara y la cross-attention dejan generar sin mirar el futuro, consultando el input. Y el contraste final: para generar hicimos 1 pasada del encoder y 4 del decoder. Al entrenar no hay loop: con la máscara, las 4 palabras salen en paralelo, en un solo forward. Cómo se entrena es lo próximo.

---

## Diapositiva 31 — Muchas gracias (≈1 min)

*[La animación arranca sola al entrar a la slide y dura unos 8 segundos. Para repetirla, volver a la 30 y avanzar.]*

Una última pasada, ahora para despedirnos. Esto es solo la capa Linear del final: a la izquierda, el vector que sale del decoder; a la derecha, algunas palabras del vocabulario. En cada paso se enciende la que gana la softmax, y esa palabra pasa a la frase de arriba: «Muchas», «gracias», «!». Y cuando gana fin de secuencia, se termina.

Muchas gracias. ¿Preguntas?

*Puntos abiertos para la revisión*
- El nombre del docente y la fecha en la diapositiva 1 son marcadores de posición.
- Idioma: diapositivas y guion en español.
