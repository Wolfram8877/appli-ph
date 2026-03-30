import flet as ft
import numpy as np
from PIL import Image
import joblib
import subprocess

# ==========================================
# CHARGEMENT DU MODÈLE
# ==========================================
print("Chargement du modèle d'IA...")
modele_ph = joblib.load('modele_knn_ph.pkl') 
print("Modèle prêt !")

# ==========================================
# 1. LE RECADRAGE
# ==========================================
def recadrer_image(chemin_image):
    image = Image.open(chemin_image).convert('RGB')
    pixels = image.load()
    l, h = image.size

    x_min, x_max = l, 0
    y_min, y_max = h, 0
    seuil = 100 

    for x in range(l):
         for y in range(h):
             r, g, b = pixels[x, y]
             if max(r,g,b) - min(r,g,b) > seuil:
                 x_min = min(x_min, x)
                 x_max = max(x_max, x)
                 y_min = min(y_min, y)
                 y_max = max(y_max, y)

    centre_x = int((x_min + x_max) / 2)
    centre_y = int((y_min + y_max) / 2)
    
    # On rogne l'image
    im_rogne = image.crop((centre_x - 150, centre_y - 150, centre_x + 150, centre_y + 150))
    return im_rogne
    
# ==========================================
# 2. LA PRÉDICTION
# ==========================================
def calculer_ph(image_decoupee):
    image_numpy = np.array(image_decoupee)
    moyenne_rgb = image_numpy.mean(axis=(0, 1))
    
    donnees_pour_prediction = moyenne_rgb.reshape(1, -1)
    prediction = modele_ph.predict(donnees_pour_prediction)
    
    return round(float(prediction[0]), 1)

# ==========================================
# 3. L'INTERFACE GRAPHIQUE (FLET sans FilePicker !)
# ==========================================
import subprocess # L'outil magique pour appeler ton Linux directement

def main(page: ft.Page):
    page.title = "Prédicteur de pH"
    page.horizontal_alignment = "center"
    page.theme_mode = "light"

    titre = ft.Text("Analyse du pH par Photo", size=30, weight="bold")
    texte_resultat = ft.Text("En attente d'une photo...", size=20)
    conteneur_image = ft.Container(width=300, height=300)

    # La nouvelle action quand on clique sur le bouton
    def on_click_choisir(e):
        try:
            # On force Linux à ouvrir la fenêtre de choix de fichier (Zenity)
            process = subprocess.run(
                ['zenity', '--file-selection', '--title=Choisir une photo'], 
                capture_output=True, 
                text=True
            )
            
            # Si tu as bien cliqué sur un fichier (code 0)
            if process.returncode == 0:
                chemin_fichier = process.stdout.strip()
                
                # 1. On affiche l'image choisie
                conteneur_image.content = ft.Image(src=chemin_fichier, width=300, height=300, fit="contain")
                texte_resultat.value = "Analyse en cours..."
                texte_resultat.color = "orange"
                page.update()

                # 2. On lance ton IA !
                image_croppee = recadrer_image(chemin_fichier)
                ph_final = calculer_ph(image_croppee)

                # 3. On affiche le résultat
                texte_resultat.value = f"Le pH estimé est : {ph_final}"
                texte_resultat.color = "green" if 6 <= ph_final <= 8 else "red"
                page.update()
                
        except Exception as erreur:
            texte_resultat.value = f"Erreur : {erreur}"
            texte_resultat.color = "red"
            page.update()

    # Un bouton tout simple, relié à notre nouvelle fonction
    bouton_photo = ft.ElevatedButton(
        "Choisir une photo", 
        on_click=on_click_choisir
    )

    # On ajoute juste les éléments visuels, sans rien de caché !
    page.add(titre, bouton_photo, conteneur_image, texte_resultat)

ft.app(target=main)
