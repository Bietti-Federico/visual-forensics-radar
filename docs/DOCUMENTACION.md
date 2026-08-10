# PDF Forensics — Documentación técnica

Plataforma de verificación forense de comprobantes PDF: un pipeline que
parsea el documento a bajo nivel, extrae features estructurales, y las
combina con reglas declarativas, detección de anomalías, un ensamble
supervisado, identificación de entidad emisora e invariantes aprendidas,
y verificación de firma digital, para producir un **risk score de 0 a 100**
con motivos explicables. Se integra a los flujos de carga de comprobantes
mediante una API HTTP (`src/pdf_forensics_api/`) con un frontend mínimo
para entrenar y verificar.

Este documento reemplaza los 12 archivos previos en `docs/` (uno por
módulo) por una versión única y acotada. El código sigue siendo la fuente
de verdad para el detalle línea por línea; acá está el *por qué* de cada
decisión de diseño.

---

## 1. Parser (`domain/pdf/`, `infrastructure/parsing/`)

Parser de PDF **propio, sin dependencias**, en vez de PyMuPDF/pikepdf.
Motivo: las librerías estándar están hechas para *renderizar* un
documento, por lo que reparan o esconden silenciosamente justo las
desviaciones que esta plataforma necesita detectar (un offset de xref
roto, un objeto redefinido fuera del mecanismo normal de actualización
incremental, un stream cuyo `/Length` declarado no coincide con su
tamaño real). Un parser forense tiene que *ver* esas desviaciones, no
esquivarlas.

**Las anomalías son datos, no excepciones.** Una estructura de PDF
malformada es entrada esperada, no un bug. `StructuralAnomaly` es un
valor de primera clase que se acumula durante el parseo. Sólo dos
excepciones reales, para archivos sin ninguna evidencia que extraer:
`NotAPdfError` (no hay header `%PDF-`) y `UnrecoverableStructureError`
(no se puede localizar ningún objeto ni por xref ni por escaneo de
fuerza bruta `N G obj`).

Algoritmo de resolución de xref/object streams (`PdfDocumentParser.parse`):
confirma el header → sigue `/Prev` fusionando cada revisión (tabla clásica
o xref stream) → si no encuentra nada, cae a un escaneo de fuerza bruta
del archivo completo → resuelve cada objeto, con recuperación puntual si
el offset declarado no coincide con lo que hay ahí → expande object
streams. Todo esto queda registrado como anomalías, nunca aborta.

**Deferido** (representado sólo como flags o dicts crudos, sin lógica):
cifrado/decifrado, firmas digitales (ver signature verification más
abajo), fuentes/imágenes/anotaciones, semántica de XMP, linearización,
filtros distintos de FlateDecode.

---

## 2. Extracción de features (`domain/features/`, `plugins/features/`)

Convierte un `PdfDocument` ya parseado en un `FeatureSet`: ~45 features
tipadas y documentadas en 10 categorías (`general`, `metadata`, `trailer`,
`catalog`, `xref`, `incremental_updates`, `objects`, `streams`,
`security`, `statistics`). Sólo lo que hoy es computable desde el modelo
de Módulo 1 — nada de resolución de fuentes/imágenes ni semántica XMP.

Cada `Feature` lleva `source` (de dónde salió exactamente) y `confidence`
(siempre `1.0` por ahora — toda feature es una lectura estructural
directa). Arquitectura de plugins explícita: `default_feature_extractors()`
en `plugins/features/__init__.py` es el único lugar a tocar para
agregar/quitar un extractor — sin auto-registro mágico, para que los
tests de cada extractor queden aislados y no dependan del orden de import.

---

## 3. Fingerprint (`domain/fingerprint/`, `application/fingerprinting/`)

Hashes SHA256 deterministas y comparables a partir del documento y sus
features: `xref_hash`, `structure_hash`, `metadata_hash`, `page_tree_hash`,
`feature_hash` (`font_hash` queda `None`, deferido — necesitaría recorrer
`/Resources/Font`). No es identificación de generador ni scoring de
riesgo — sólo hashes comparables. Deliberadamente no lee los nombres de
features de Módulo 2 por string (para no acoplarse a un typo silencioso);
vuelve a resolver `/Info` directamente desde `PdfDocument`.

---

## 4. Motor de reglas (`domain/rules/`, `application/rule_engine/`, `plugins/rules/`)

Reglas declarativas e independientes sobre un `FeatureSet` ya extraído:
cada una devuelve `severity`/`confidence`/`explanation`/`references` o
`None` si no aplica. Nunca "adivinan" — si falta el dato necesario,
devuelven `None`, no un finding de baja confianza.

| Regla | Dispara cuando | Severidad |
|---|---|---|
| `creation_after_mod_date` | `/CreationDate` posterior a `/ModDate` | WARNING |
| `pdf_version_mismatch` | versión del header ≠ `/Root/Version` | INFO (es un override válido del spec) |
| `missing_info_dictionary` | no hay `/Info` | WARNING |
| `duplicate_object_ids` | ids de objeto duplicados | WARNING |
| `broken_xref` | anomalías de xref presentes | CRITICAL/WARNING |
| `unexpected_incremental_update` | más de 2 revisiones | WARNING |
| `trailer_missing_root` | no hay `/Root` en el trailer | WARNING |

**`entity_template_mismatch`** ya no es una regla-plugin hardcodeada: se
explica en la sección 8 (Invariantes por entidad), porque necesita estado
ajustado por entidad y se invoca directamente desde `ScoreDocumentUseCase`,
no desde `default_rules()`.

---

## 5. Detección de anomalías (`domain/anomaly_detection/`, `plugins/anomaly_detection/`)

Cuatro detectores **no supervisados** — Isolation Forest, One-Class SVM,
Local Outlier Factor y un autoencoder chico en PyTorch — que puntúan qué
tan raro es un documento respecto de un lote de referencia con el que
fueron ajustados. Nunca ven una etiqueta de fraude, sólo `FeatureSet`s.

Sólo entran al vector features `INTEGER`/`FLOAT`/`BOOLEAN` (histogramas y
strings quedan afuera); un valor `None` se omite, no se fuerza a `0.0`
("desconocido" y "cero" no son lo mismo). Convención de signo: **más alto
siempre es más anómalo**, incluso cuando cada algoritmo internamente
funciona al revés — cada detector normaliza su propio score para cumplir
esa convención.

**Importante (ver sección 9):** ajustar esto de forma global mezclando
entidades distintas es engañoso — hay que ajustarlo por entidad.

---

## 6. Ensamble de ML (`domain/ml_ensemble/`, `plugins/ml_ensemble/`)

Contraparte **supervisada** de la anterior: seis clasificadores entrenados
con documentos etiquetados genuino/manipulado. Sólo se implementó
**XGBoost** como gradient boosting (no CatBoost/LightGBM — decisión
explícita, se solapan funcionalmente). Objetivo binario: `is_original`;
la clase positiva es "manipulado", así que `probability` es siempre
P(manipulado) — "más alto = más sospechoso", consistente con Módulo 5.

Cuatro modelos base con SHAP real (`random_forest`, `extra_trees`,
`logistic_regression`, `xgboost`) y dos ensambles (`stacking_ensemble`,
`voting_ensemble`) sin SHAP — un combinador heterogéneo necesitaría
`KernelExplainer`, deferido por lento y poco confiable sin tuning fino.

**Requiere ejemplos reales de fraude confirmado para entrenar de forma
honesta** (ver sección 9 — por eso su activación está gateada por entidad).

---

## 7. Identificación de entidad (`domain/entity_identification/`, `plugins/entity_identification/`)

Clasificador multiclase (`RandomForestEntityClassifier`, uno solo — el
corpus real no justifica un ensamble) que predice qué entidad conocida
(ANSES, Municipalidad de La Rioja, Municipalidad de Jujuy, y cualquier
otra que se cargue) probablemente emitió el documento, usando features
numéricas más el string crudo de `/Producer`/`/Creator`. Alimenta el
componente `entity_consistency` del Risk Score (`1 - confianza`) y el
finding `entity_template_mismatch`.

**Qué NO es:** el brief original pide comparar lo que un documento *dice
ser* (su contenido visible) contra lo que *estructuralmente* es — esta
plataforma no tiene extracción de texto de página todavía, así que no
puede leer qué institución dice el documento y compararlo con la
predicción estructural. Lo implementado es más angosto pero real:
"¿la estructura de este documento coincide con confianza con alguna de
las plantillas reales que el clasificador vio?"

### Hiperparámetros ajustados para lotes chicos, no los de sklearn por defecto

`RandomForestClassifier` con `bootstrap=False` y `max_features=None`, no
los valores por defecto. Con sólo 2-3 documentos genuinos por entidad, el
resampleo con reemplazo (`bootstrap=True` por defecto) deja a algunos
árboles del ensamble sin ver ningún ejemplo de una clase, y el sorteo de
features por split (`max_features="sqrt"` por defecto) puede excluir por
pura casualidad la señal casi perfecta que sí tenemos (`/Producer`,
codificado one-hot en unas pocas columnas). Verificado por
leave-one-out sobre el corpus real: con los valores por defecto, algunos
documentos genuinos daban apenas 0.5-0.6 de confianza (y en una prueba
similar, un documento terminó mal clasificado); con `bootstrap=False` y
`max_features=None`, los mismos documentos dieron ≥0.88 sin ningún error.
Cada árbol ve todas las muestras y todas las features disponibles — con
tan pocos datos, la ganancia de reducir varianza vía muestreo aleatorio
no compensa el ruido que introduce.

---

## 8. Explicabilidad (`domain/explainability/`, `application/explainability/`)

Agregador puro: toma los resultados de reglas, anomalías y ML, y produce
"motivos" en texto plano — nunca un motivo inventado. Cada string sale de
algo realmente calculado (`RuleFinding.explanation`, un `AnomalyScore` con
`is_anomaly=True`, una predicción positiva de ML). Las top-features SHAP
se mantienen separadas por modelo, porque `TreeExplainer` y
`LinearExplainer` operan en espacios distintos — mezclarlas implicaría una
comparabilidad que no existe.

---

## 9. Risk Report (`domain/risk/`, `application/risk_report/`)

Combina los siete componentes en un score final de 0 a 100:
`round(clamp(Σ(score·peso), 0, 1) * 100)`.

| Componente | Fórmula | Por qué |
|---|---|---|
| `rule_engine` | Noisy-OR sobre severidades de reglas | La evidencia independiente satura hacia 1.0 sin que una suma simple se pase |
| `ml_probability` | Promedio de probabilidades del ensamble | Todas son P(manipulado), comparables entre sí |
| `anomaly_detection` | Fracción de detectores que marcaron `is_anomaly` (no el score crudo, no Noisy-OR) | El score crudo no es comparable entre los 4 detectores; el flag sí. Un solo detector en el límite (ej. one_class_svm con score≈0, artefacto conocido con pocas muestras por entidad) pesa proporcionalmente menos que varios de acuerdo, en vez de contribuir lo mismo que un consenso real |
| `structural` | Densidad de anomalías / cantidad de objetos | Señal general, distinta de los patrones específicos del motor de reglas |
| `metadata` | Fracción de campos de `/Info` faltantes | Metadata faltante correlaciona con "limpieza" del documento |
| `entity_consistency` | `1 - confianza` de identificación de entidad | Baja confianza en calzar con alguna plantilla conocida es señal |
| `signature_integrity` | Peor hallazgo entre las firmas embebidas | Ver sección 10 |

Los pesos (`RiskWeights`) son un punto de partida documentado, no un
modelo calibrado — no hay todavía dataset etiquetado suficiente para
ajustarlos empíricamente. `RiskWeights.without_ml_probability()` anula el
peso de `ml_probability` y reescala el resto a que sigan sumando 1.0,
para cuando una entidad todavía no tiene ML Ensemble activado.

### Por qué el ajuste es **por entidad**, no global

Ajustar Anomaly Detection globalmente mezclando ANSES/La Rioja/Jujuy es
activamente engañoso: con sólo los documentos reales disponibles, marcó
6 de 10 documentos genuinos como anómalos, porque entidades
estructuralmente distintas se hacen parecer "anómalas" entre sí al
mezclarlas. Ajustar **por entidad** (un documento de ANSES sólo se compara
contra otros de ANSES) elimina casi todo ese ruido.

ML Ensemble tiene el mismo problema más agudo: necesita ejemplos
etiquetados de manipulación real, y hay muy pocos casos de fraude
confirmado. Entrenarlo con variantes sintéticas fabricadas no aporta señal
real — enseña a reconocer la técnica de fabricación, no el fraude. Por
eso el flujo de reentrenamiento de producción nunca usa datos sintéticos
(eso queda para el proyecto hermano `pdf-forensics-benchmark`, sólo I+D) y
en cambio **activa el ML Ensemble automáticamente, por entidad**, cuando
se cruzan umbrales mínimos de documentos genuinos y de fraude confirmado:

```python
# application/model_training/retrain_models_use_case.py
ANOMALY_MIN_GENUINE_PER_ENTITY = 2
ML_ENSEMBLE_MIN_GENUINE_PER_ENTITY = 5
ML_ENSEMBLE_MIN_CONFIRMED_FRAUD_PER_ENTITY = 3
```

---

## 10. Verificación de firma digital (`domain/signature_verification/`, `infrastructure/signature_verification/`)

Valida firmas PKCS#7/CMS embebidas: si el digest sigue coincidiendo con
los bytes firmados (`digest_intact`), si la firma en sí es válida
(`cryptographically_valid`), y si cubre todo el archivo o quedó contenido
agregado después de firmar (`coverage`). Usa la librería
[pyHanko](https://github.com/MatthiasValvekens/pyHanko) en vez de parsear
esto a mano — a diferencia del parser de Módulo 1, ASN.1/PKCS#7/firmas
criptográficas es exactamente el tipo de parsing sensible que no conviene
reimplementar.

**Validación de cadena de confianza excluida a propósito**: no hay una
CA raíz configurada (ej. ONTI para certificados de AFIP/ANSES), y reportar
"no confiable" sin una raíz configurada marcaría en rojo cualquier firma
genuina — peor que no reportarlo.

Scoring (peso 0.10 del Risk Score, gana el peor hallazgo entre todas las
firmas): sin firma = `0.0` (no es sospechoso por sí solo); digest roto =
`1.0` (la señal de manipulación más fuerte posible); cobertura parcial =
`0.7`; firma inválida = `0.9`; firma válida = `0.0`.

### Bug de event loop (corregido)

pyHanko valida firmas de forma async internamente y usa su propio
`asyncio.run(...)`, que **falla si ya hay un event loop corriendo** — como
pasa siempre dentro de un handler async de FastAPI. Ese error quedaba
absorbido en silencio por un `except Exception` pensado para firmas
individuales rotas, así que **todas las firmas desaparecían al verificar
por la API**, aunque funcionaran perfecto desde un script plano. Se
corrigió llamando a la API async de pyHanko directamente y corriéndola en
un thread dedicado cuando ya hay un loop activo; además, todas las firmas
de un documento se validan en un solo `asyncio.gather(...)` en vez de un
thread/loop nuevo por cada firma (evita el mismo problema repetido y el
costo de crear un executor por firma en documentos con múltiples firmas).

---

## 11. Invariantes aprendidas por entidad (`application/entity_invariants/`)

Reemplaza lo que antes era un diccionario **hardcodeado a mano** de
invariantes estructurales por entidad (ej. "ANSES siempre tiene
`/AcroForm`", "La Rioja siempre embebe exactamente 1 JPEG") — cada entidad
nueva requería que alguien abriera el código y agregara una entrada a
mano. Ahora se **minan automáticamente en cada reentrenamiento**: toda
feature booleana, o toda clave de un histograma (ej.
`streams.filter_histogram`), cuyo valor sea idéntico en todos los
documentos genuinos de una entidad, se vuelve un `LearnedInvariant`.

Deliberadamente **no** se minan features numéricas continuas (tamaño en
bytes, cantidad de objetos, etc.) — varían con el contenido legítimo del
documento, así que una coincidencia en un lote chico se convertiría en
falsos positivos constantes apenas creciera el corpus.

Umbral mínimo de muestras: `TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY = 1`
— se confía en un solo documento confirmado como real de inmediato en vez
de esperar volumen que puede no existir todavía. Costo conocido y
aceptado: con `n=1`, features incidentales de ese único documento (no sólo
las genuinamente estructurales) también quedan como "invariantes", así
que hay que esperar más ruido de falsos positivos en las primeras cargas
de esa entidad hasta que más documentos descarten las coincidencias. Como
las invariantes se recalculan **desde cero en cada retrain** (no son
incrementales), se autocorrigen solas a medida que entra más volumen.

Al verificar, produce el mismo `RuleFinding` (mismo `rule_id`,
`entity_template_mismatch`) que la regla vieja, con una explicación
generada automáticamente a partir de los valores esperado/real en vez de
texto escrito a mano. La severidad ya no es siempre `WARNING`: si más de
la mitad de las invariantes de la entidad quedan contradichas a la vez,
sube a `CRITICAL` — contradecir casi todo lo que esa entidad siempre
cumple no es "un poco distinto", es evidencia fuerte de que el documento
no responde a esa plantilla en absoluto.

---

## 12. Ingesta de datos de entrenamiento — sólo I+D (`domain/training_data/`, `infrastructure/benchmark_import/`)

Este flujo lee el output CSV (`labels.csv` + `hashes.csv`) del proyecto
hermano `pdf-forensics-benchmark`, que genera PDFs sintéticos (y reales,
por decisión explícita) con variantes de transformación y ground truth,
y los re-featuriza con el pipeline **propio** de esta plataforma (no el
feature space del benchmark). Sirve para pruebas de humo end-to-end contra
datos sintéticos — el flujo de producción real (sección 13) **no** usa
nada de esto.

---

## 13. API de verificación + frontend de entrenamiento (`src/pdf_forensics_api/`)

Punto de integración a los flujos de carga de comprobantes. Tres reglas
de diseño:

1. **Verificar nunca escribe a disco.** `/verify` procesa los bytes
   enteramente en memoria; la respuesta JSON es el único output — sin
   logs ni archivos por request.
2. **El disco sólo crece cuando alguien agrega deliberadamente un
   archivo de entrenamiento.** Un solo `model_store.joblib`, sobreescrito
   en cada retrain — sin historial de versiones.
3. **Reentrenar es una acción manual y explícita.** Subir un archivo de
   entrenamiento nunca dispara un retrain solo — sólo agrega el archivo a
   `training_corpus/`. Un `POST /retrain` separado relee todo el corpus.

### Corpus de entrenamiento: carpetas, sin manifiesto

```
training_corpus/
  genuine/
    ANSES/*.pdf
    LA_RIOJA/*.pdf
    <ENTIDAD_NUEVA>/*.pdf     # agregar una entidad = una carpeta nueva, cero cambios de código
  confirmed_fraud/
    ANSES/*.pdf
    ...
```

El nombre de la carpeta **es** la etiqueta de entidad. Agregar una entidad
nueva no requiere ningún cambio de código para Identificación de Entidad,
Anomaly Detection, ML Ensemble ni Invariantes — todos se ajustan por
entidad a partir de lo que haya en disco.

### Esquema de persistencia (v3.0.0)

```python
{
  "schema_version": "3.0.0",
  "entity_classifiers": {...},   # un clasificador global de entidad
  "per_entity": {
    "ANSES": {
      "detectors": (...) | (),    # () si genuine_count < ANOMALY_MIN_GENUINE_PER_ENTITY
      "models": (...) | (),       # () si ML Ensemble no está listo
      "invariants": (...) | (),   # () si genuine_count < TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY
      "genuine_count": 4,
      "confirmed_fraud_count": 1,
      "ml_ensemble_ready": False,
    },
    ...
  },
}
```

Cada cambio de versión (`1.0.0`→per-entidad, `2.0.0`→invariantes) es una
ruptura deliberada, sin migración — hay que reentrenar desde el corpus.

### `ScoreDocumentUseCase`: un único camino de scoring

Compartido entre `/verify` y los scripts de CLI, para que no diverjan.
Dado `pdf_bytes` y el bundle ya cargado: parsea + extrae features +
fingerprint → identifica entidad → busca los detectores/modelos/
invariantes de esa entidad en el bundle (si no hay bundle, todo queda
vacío — "nada detectado", no un error) → motor de reglas + chequeo de
invariantes + verificación de firma → arma el Risk Report con los pesos
completos o reducidos según si el ML Ensemble está listo para esa
entidad.

### Endpoints

| Endpoint | Qué hace |
|---|---|
| `GET /` | Sirve el frontend estático. |
| `POST /verify` | Sube un PDF, corre todo el pipeline en memoria, devuelve el JSON completo **en español** (keys y textos — ver abajo). `422` si no es un PDF legible. |
| `POST /training-data/suggest-entity` | Corre el clasificador actual sobre un archivo y sugiere la entidad más probable — para precargar el campo al subir un documento nuevo, sin que alguien tenga que saber (o escribir bien) el nombre de la entidad. |
| `POST /training-data/genuine` | Sube un documento genuino a `training_corpus/genuine/<entidad>/`. No reentrena. |
| `POST /training-data/confirmed-fraud` | Igual, bajo `confirmed_fraud/`. |
| `GET /training-data/summary` | Conteos agregados por entidad. |
| `GET /training-data/files` | Listado de archivos individuales (entidad, tipo, nombre). |
| `DELETE /training-data/genuine/{entidad}/{archivo}` | Elimina un archivo genuino. `404` si no existe. |
| `DELETE /training-data/confirmed-fraud/{entidad}/{archivo}` | Igual, para fraude confirmado. |
| `POST /retrain` | Reentrena todo el corpus, sobreescribe el modelo, recarga en memoria — sin reiniciar el proceso. |

**Seguridad de rutas**: tanto la entidad como el nombre de archivo se
sanitizan (`_sanitize_path_component`) antes de usarse como segmentos de
ruta, para que no se pueda escapar del directorio del corpus. **Tamaño
máximo de subida**: 25 MB.

### El JSON de `/verify` está en español

El dominio y la lógica interna (nombres de clases, features, motor de
reglas) siguen en inglés — es lo que leen los tests, los scripts de CLI y
el resto de este documento. Pero el límite de la API es el único lugar
donde eso importa a un humano: `pdf_forensics_api/serialization.py`
traduce todo lo que cruza esa frontera, tanto las *keys* como los textos
generados, ya que los usuarios reales de este despliegue hablan español.
Forma del JSON de `/verify`:

```json
{
  "huella_digital": { "generador": ..., "productor": ..., "hash_estructura": ..., ... },
  "entidades_predichas": [
    { "clasificador": ..., "entidad_predicha": "ANSES", "confianza": 0.91, "probabilidades": {...} }
  ],
  "firmas": [
    { "campo": ..., "digest_integro": true, "criptograficamente_valida": true,
      "cobertura": "archivo_completo" | "parcial" | "no_determinada",
      "firmante": ..., "fecha_firma": ... }
  ],
  "puntaje_riesgo": 21,
  "componentes": [
    { "nombre": "motor_de_reglas" | "probabilidad_ml" | "deteccion_de_anomalias" |
               "estructural" | "metadatos" | "consistencia_de_entidad" | "integridad_de_firma",
      "puntaje": 0.0, "peso": 0.31 }
  ],
  "motivos": ["Identificado como ANSES (confianza=91%), pero este documento contradice..."],
  "caracteristicas_principales": [ { "modelo": ..., "caracteristica": ..., "valor_shap": ... } ]
}
```

Los `motivos` (explicaciones de reglas, hallazgos de anomalía, predicciones
de ML) se generan directamente en español en sus fuentes
(`plugins/rules/*.py`, `check_entity_invariants_use_case.py`,
`GenerateExplanationUseCase`) — no es una traducción de texto en inglés
sobre la marcha. Los demás endpoints (`/training-data/*`, `/retrain`)
siguen con keys en inglés por ahora; sólo `/verify` (lo que efectivamente
ve el usuario final del sistema de carga de comprobantes) se tradujo.

**Dos puntos que no siempre traen un motivo en texto**: los componentes
`metadatos` y `estructural` son fórmulas continuas (fracción de campos de
`/Info` faltantes; densidad de anomalías), no hallazgos binarios — pueden
sumar puntos al `puntaje_riesgo` sin aparecer como una frase en `motivos`,
que sólo junta explicaciones de reglas disparadas, detectores de anomalía
marcados o predicciones de ML positivas.

### Configuración (variables de entorno)

| Variable | Default |
|---|---|
| `PDF_FORENSICS_TRAINING_CORPUS_DIR` | `training_corpus` |
| `PDF_FORENSICS_MODEL_STORE_PATH` | `model_store.joblib` |
| `PDF_FORENSICS_LOG_PATH` | `pdf_forensics_api.log` |

### Logging

Un archivo de log rotativo (5 MB × 3 backups), configurado al arrancar el
proceso (no al importar el módulo, para que la variable de entorno se
pueda pisar en tests). Cada endpoint loguea su resultado (subida
guardada, archivo borrado, resultado de verificación, resumen de
reentrenamiento) en español. Cualquier excepción no controlada queda
capturada por un handler global que loguea el traceback completo antes de
devolver un `500` genérico — un error de producción siempre queda en el
log aunque nada más lo muestre.

### Frontend (`static/index.html`)

Un solo archivo HTML+JS vanilla, sin build, servido por el mismo backend.
Tres secciones: **Verificar** (drag-and-drop, muestra el resultado
legible y el JSON crudo completo en un `<details>`, para depurar
exactamente lo que recibiría un sistema consumidor); **Corpus** (tabla
agregada por entidad + tabla de archivos individuales con botón "Quitar");
**Agregar documento de entrenamiento** (la entidad es un campo de texto
libre, no un dropdown fijo, pero se precompleta automáticamente al elegir
el archivo vía `suggest-entity` — editable, para una entidad nueva o una
sugerencia de baja confianza).

### Ejecución

```bash
poetry run uvicorn pdf_forensics_api.app:app --host 0.0.0.0 --port 8000
```

---

## Convenciones generales del código

- **Arquitectura limpia**: `domain/` (tipos puros) → `application/` (casos
  de uso, sólo dependen de `domain/` y de *ports* `Protocol`) →
  `infrastructure/`/`plugins/` (implementaciones concretas, dependen hacia
  adentro, nunca al revés).
- **Plugins con wiring explícito**, no auto-registro: `default_*()` en
  cada `plugins/*/__init__.py` es el único lugar a tocar.
- **Las anomalías/fallas son datos, no excepciones**, salvo cuando
  genuinamente no hay evidencia que extraer.
- **Sin datos sintéticos en el flujo de producción** — sólo en el
  proyecto hermano de I+D.
- **Comentarios explican el *por qué*, no el *qué*** — el código ya dice
  qué hace si está bien nombrado.
