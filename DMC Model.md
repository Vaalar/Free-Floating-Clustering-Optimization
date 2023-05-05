# DMC Model Explanation

## Etapa de inicialización 
Se leen los valores del archivo de datos y se cargan en el programa. Este archivo contiene las coordenadas de de los objetos que se quieren clusterizar y su clase. La estructura de este archivo es de 3 columnas (X, Y, Clase del Punto).

## Etapa de clustering
Realiza la clusterización de los objetos utilizando la estrategia K-Means++. La clusterización final escogida, es la óptima de n configuraciones de clústeres diferentes calculadas. Esto se mide utilizando el índice de Calinski-Harabasz, que mide la cohesión de los objetos en cada clúster y de todos los clústeres. A mayor valor del índice, mayor cohesión presenta la configuración de clústeres.  

Se puede dividir en 2 etapas:

1. `Inicialización de los centroides`:  
    Cáculo de los centroides iniciales:
    1. Se calcula la media de todos los valores y se escoge el punto más alejado de ella como primer centroide. 

    2. Se escoge el punto más alejado del centroide previamente calculado y se asigna como segundo centroide.

    3. Por último, se calcula el punto más alejado de todos los centroides y se escoge como centroide.  

    4. Una vez calculados los centroides, se procede a asignar los objetos al centroide más cercano.  
    
    El paso 3 se repite hasta haber creado tantos centroides como clústeres se deseen tener.

    Los centroides calculados previamente se utilizan sólo para la inicialización de los clústeres. Una vez estos ya se han hallado, los centroides pasan a ser la media de los objetos que conforman cada clúster, por lo que, al haber variado la posición de los centroides, se tienen que volver a calcular los clústeres ya que pueden haber objetos que estén ahora más cerca de otro centroide. Este recálculo, se realiza hasta que ningún punto se reasigna de clúster.

    Cálculo de los clústeres:  
    
    5. Recálculo de centroides.

    6. Reasignación de los objetos 

2. `Clustering`:  
Una vez hallados la configuración clústeres, se proceden a repetir los pasos 1 y 2 hasta haber calculado las n configuraciones deseadas. Una vez halladas, se escoge aquella configuración con mayor índice de Calinski-Harabasz y se pasa a la siguiente etapa del flujo.

## Etapa de visualización
Se representa la configuración de clústeres en un mapa. Si algún punto que se movió cambió de clúster, se mostrará la posición en el instante t-1 con un rombo negro y su posición actual en el instante t mediante un círculo con el color de los objetos de su clúster. Ambas figuras de representación del punto que se reasignó de clúster están unidas por una línea para mejor seguimiento visual. A su vez, los centroides de cada clúster están representados mediante una X de color rojo.

## Etapa de evaluación de finalización
Se evalúa sí se calculó la configuración de clústeres del último instante de tiempo. Si era la última, se sale de la ejecución del algoritmo. Si no era la última, se procede a la etapa de evolución.

## Etapa de evolución
Se actualiza la posición de los objetos con las coordenadas de ellos en el instante t+1. Consiste en un flujo de datos continuo en el cuál tenemos las coordenadas de cada objeto en cada instante de tiempo.

## Etapa de evaluación de reasignación
Si un punto se mueve lo suficiente, éste puede quedar más cerca de otro centroide, por lo que habría que recalcular la configuración de clústeres. Esto requiere una carga computacional muy elevada ya que hay que calcular n configuraciones de clústeres. Para arreglar este problema, se introduce la reasignación de clúster, que tiene en cuenta cuántos objetos se movieron y ahora están más cerca de otro centroide. En función de esa cantidad, decide si se ha de recalcular o no la configuración de clústeres para el instante t+1.

Si la cantidad de objetos que se movieron y ahora están más cerca de otro centroide supera un cierto umbral, se considera que la configuración de clústeres podría haber cambiado de forma notoria, por lo que se procedería a volver a la etapa de clustering para calcular la nueva configuración de clústeres.  

Sin embargo, si la cantidad de objetos que se movieron y ahora están más cerca de otro centroide es menor que el umbral de reclusterización, entonces estos objetos simplemente se asignarían a otro clúster y se pasaría directamente a la etapa de visualización, ya que se considera que el cambio no es lo suficientmente notorio como para merecer la pena una reclusterización.
