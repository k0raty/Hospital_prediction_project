#PATH="/home/k0raty/Documents/Helean/database/"
PATH = 'C:\\Users\\anton\\Documents\\Helean\\database\\'
     

SPECIAL_CMD = [
    14,  # maternité
    15,  # néonatologie
    28,  # séances
    90   # autres
]

UM_LIST = [
    'AMBU',
    '0042',
    '0036',
    '0003',
    '0002',
    'GENE',
    '0011',
    '0041',
    'UROL',
    '0019',
    '0004',
    '0001',
    '0014',
    '0018',
    'ORTH',
    'GAST',
    '0028',
    '0024',
    '0021',
    'GYNE',
    'ORL ',
    '0005',
    '0008',
    '0020',
    '0010',
    '0012',
    '0022',
    'ZTCD',
    'ZCDE',
    'YGEN',
    '0015',
    '0050',
    '0054',
    'OPHT',
    'YCAR',
    'PLAS',
    '0055',
    '0016',
    '0017',
    '0013'
]

CMD = {
    1:    "Affections du système nerveux",
    2:    "Affections de l'œil",
    3:    "Affections des oreilles, du nez, de la gorge, de la bouche et des dents",
    4:    "Affections de l'appareil respiratoire",
    5:    "Affections de l'appareil circulatoire",
    6:    "Affections du tube digestif",
    7:    "Affections du système hépatobiliaire et du pancréas",
    8:    "Affections et traumatismes de l'appareil musculosquelettique et du tissu conjonctif",
    9:    "Affections de la peau, des tissus sous-cutanés et des seins",
    10:   "Affections endocriniennes, métaboliques et nutritionnelles",
    11:   "Affections du rein et des voies urinaires",
    12:   "Affections de l'appareil génital masculin",
    13:   "Affections de l'appareil génital féminin",
    14:   "Grossesses pathologiques, accouchements et affections du post-partum",
    15:   "Nouveau-nés, prématurés et affections de la période périnatale",
    16:   "Affections du sang et des organes hématopoïétiques",
    17:   "Affections myéloprolifératives et tumeurs de siège imprécis ou diffus et/ou CMA",
    18:   "Maladies infectieuses et parasitaires",
    19:   "Maladies et troubles mentaux",
    20:   "Troubles mentaux organiques liés à l'absorption de drogues ou induits par celles-ci",
    21:   "Traumatismes, allergies et empoisonnements",
    22:   "Brûlures",
    23:   "Facteurs influant sur l'état de santé et autres motifs de recours aux services de santé",
    24:   "Séjours de moins de 2 jours",
    25:   "Maladies dues à une infection par le VIH",
    26:   "Traumatismes multiples graves",
    27:   "Transplantations d'organes",
    28:   "Séances",
    90:   "Erreurs et autres séjours inclassables",
}

COMPLEXITE_GHM = {
    "1":    "niveau de sévérité 1",
    "2":    "niveau de sévérité 2",
    "3":    "niveau de sévérité 3",
    "4":    "niveau de sévérité 4",
    "A":    "niveaux de sévérité pour les classes CMD 14 et CMD 15",
    "B":    "niveaux de sévérité pour les classes CMD 14 et CMD 15",
    "C":    "niveaux de sévérité pour les classes CMD 14 et CMD 15",
    "D":    "niveaux de sévérité pour les classes CMD 14 et CMD 15",
    "T":    "très courte durée",
    "J":    "ambulatoire",
    "E":    "décès de courte durée",
    "Z":    "non séquencé"
}

TYPE_GHM = {
    "C":    "groupe chirurgical avec acte classant",
    "K":    "groupe avec acte classant non opératoire",
    "M":    "groupe médical sans acte classant",
    "Z":    "groupe indifférencié avec ou sans acte classant opératoire"
}

SEXE = {
    1:    "Homme",
    2:    "Femme"
}

MODE_ENTREE = {
    6:    "mutation",
    7:    "transfert",
    8:    "domicile"
}

PROVENANCE = {
    1:    "provenance d’une unité du MCO sauf unité de réanimation",
    2:    "provenance d'une unité de soins de suite et de réadaptation",
    3:    "provenance d'une unité de soins de longue durée",
    4:    "provenance d'une unité de psychiatrie",
    5:    "provenance du domicile",
    7:    "provenance d'une structure d'hébergement médicosociale"
}

MODE_SORTIE = {
    6:    "mutation",
    7:    "transfert définitif",
    8:    "retour au domicile",
    9:    "décès"
}


DESTINATION = {
    1:    "vers une autre unité du MCO",
    2:    "vers une unité de soins de suite et de réadaptation",
    3:    "vers une unité de soins de longue durée",
    4:    "vers une unité de psychiatrie",
    6:    "vers l’hospitalisation à domicile",
    7:    "vers une structure d'hébergement médicosociale"
}
