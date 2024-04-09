
kobieta(X) :- osoba(X), \+ mezczyzna(X).

ojciec(X,Y) :- rodzic(X,Y), mezczyzna(X).

matka(X,Y) :- rodzic(X,Y), kobieta(X).

corka(X,Y) :- rodzic(Y,X), kobieta(X).

brat_rodzony(X,Y) :- mezczyzna(X), osoba(Y), rodzic(Z,X), rodzic(Z,Y), X \= Y.

brat_przyrodni(X,Y) :- mezczyzna(X), osoba(Y), rodzic(Z,X), rodzic(Z,Y), rodzic(W,X), rodzic(W,Y), Z \= W, X \= Y.

kuzyn(X,Y) :- rodzic(Z,X), rodzic(W,Y), brat_rodzony(Z,W).

dziadek_od_strony_ojca(X,Y) :- mezczyzna(X), ojciec(X,Z), ojciec(Z,Y).

dziadek_od_strony_matki(X,Y) :- mezczyzna(X), matka(Z,Y), rodzic(X,Z).

dziadek(X,Y) :- mezczyzna(X), rodzic(X,Z), rodzic(Z,Y).

babcia(X,Y) :- kobieta(X), rodzic(X,Z), rodzic(Z,Y).

wnuczka(X,Y) :- kobieta(Y), rodzic(Z,Y), rodzic(X,Z).

przodek_do2pokolenia_wstecz(X,Y) :- rodzic(X,Z), rodzic(Z,Y).

przodek_do3pokolenia_wstecz(X,Y) :- rodzic(X,A), rodzic(A,Z), rodzic(Z,Y).
