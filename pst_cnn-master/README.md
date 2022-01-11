>  
>  
>  Ces scripts permettent de réaliser l'apprentissage et le test d'un data set,
pour la reco d'activités humaines.

Le data set est composé d'un folder train et test.
Chacun contient 9 fichiers X en input et un fichier Y output.
Les 9 fichiers en input sont : 
les valeurs de l'accèléromètre (X,Y,Z)
les valeurs du gyroscope (X,Y,Z)
les valeurs de l'accèléromètre sans la gravité terrestre (X,Y,Z)

Les fichiers sont composer de la manière suivante : 
Une sliding window de 128 valeurs est appliquées, il y a donc 128 valeurs par ligne

Chaque ligne correspond alors à une activité, 
représentée dans le fichier Y par un chiffre qui renvoi à une activité particulière.

1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING

` test `
| header | header |
| ------ | ------ |
| cell | cell |
| cell | cell | 