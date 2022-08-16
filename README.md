# ictqt
Repository with codes for my internship at IQCTQ in Gdańsk


Kilka uwag (ew. pytań):
Główny plik to bb84.py, binary.py ma sam algorytm BINARY, który wygodniej mi było zrobić osobno i importować do bb84.py;
Wszystkie testy robiłem z random seed 261135, moim nr indeksu na studiach;
Zgodnie ze wzorem z książki liczę quantum gain jako iloczyn tych wszystkich parametrów, chociaż to dziwne, że jak chcę mieć stratę 10% informacji, to samo additional loss ustawiam na 0.9. W każdym razie testowałem cały program na quantum gain w wysokości 1 lub 0.9 (mam to u góry w funkcji qc_gain, masz jakieś sugestie, jak to lepiej zaimplementować?);
Przy quantum gain 0.9 i prawdopodobieństwie zmiany 0.1 na 20 bitach oryginalnej wiadomości Alicji wychodzi już z siftingu klucz z jednym błędem;
Dalsze etapy testowałem już na 100 bitach oryginalnej wiadomości;
Nie zapoznałem się jeszcze z różnymi strategiami ataków - czy mam szybko zmienić sposób generowania błędów?
Na razie mam ten program prosty, bo zależało mi na łatwym debugowaniu. Jeśli wszystko jest ok, to przerzucę go jeszcze w programowanie obiektowe, żeby stworzyć klasę QKD, jej atrybuty (stringi baz, bitów itd.) oraz metody (pomiary Boba, sifting itd.), żeby było elegancko oraz można było wygodnie uruchamiać np. wiele rund generowania klucza z różnymi parametrami i robić numeryczne eksperymenty na tych algorytmach.
