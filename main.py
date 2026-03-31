import flet as ft
import numpy as np
from PIL import Image
import joblib
import os

# Création du dossier temporaire pour les images du téléphone
if not os.path.exists("uploads"):
    os.makedirs("uploads")

print("Chargement de l'IA sur le serveur...")
modele_ph = joblib.load('modele_knn_ph.pkl')

def recadrer_image(chemin_image):
    image = Image.open(chemin_image).convert('RGB')
    pixels = image.load()
    l, h = image.size
    x_min, x_max, y_min, y_max = l, 0, h, 0
    seuil = 100 

    for x in range(l):
         for y in range(h):
             r, g, b = pixels[x, y]
             if max(r,g,b) - min(r,g,b) > seuil:
                 x_min, x_max = min(x_min, x), max(x_max, x)
                 y_min, y_max = min(y_min, y), max(y_max, y)

    centre_x, centre_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    return image.crop((centre_x - 150, centre_y - 150, centre_x + 150, centre_y + 150))

def calculer_ph(image_decoupee):
    image_numpy = np.array(image_decoupee)
    moyenne_rgb = image_numpy.mean(axis=(0, 1)).reshape(1, -1)
    return round(float(modele_ph.predict(moyenne_rgb)[0]), 1)

def main(page: ft.Page):
    page.title = "IA pH Analyzer"
    page.horizontal_alignment = "center"
    page.theme_mode = "light"

    titre = ft.Text("Testeur de pH par IA", size=30, weight="bold")
    texte_resultat = ft.Text("En attente d'une photo...", size=20)
    conteneur_image = ft.Container(width=300, height=300, border=ft.border.all(1, "grey"), border_radius=10)

    def on_upload(e: ft.FilePickerUploadEvent):
        if e.progress == 1.0: 
            chemin_fichier = os.path.join("uploads", e.file_name)
            conteneur_image.content = ft.Image(src=f"/{e.file_name}", width=300, height=300, fit="contain")
            texte_resultat.value = "Analyse IA en cours..."
            page.update()

            try:
                img = recadrer_image(chemin_fichier)
                ph = calculer_ph(img)
                texte_resultat.value = f"pH ESTIMÉ : {ph}"
                texte_resultat.color = "green" if 6 <= ph <= 8 else "red"
            except Exception as err:
                texte_resultat.value = f"Erreur : {err}"
            page.update()

    def on_result(e: ft.FilePickerResultEvent):
        if e.files:
            texte_resultat.value = "Envoi de la photo au serveur..."
            page.update()
            selecteur.upload([
                ft.FilePickerUploadFile(e.files[0].name, upload_url=page.get_upload_url(e.files[0].name, 60))
            ])

    selecteur = ft.FilePicker(on_result=on_result, on_upload=on_upload)
    page.overlay.append(selecteur)

    bouton = ft.ElevatedButton("Prendre une photo", on_click=lambda _: selecteur.pick_files())

    page.add(titre, bouton, conteneur_image, texte_resultat)

# LA LIGNE CLÉ POUR LE CLOUD RENDER :
# On récupère le port donné par Render, sinon on utilise 8000 par défaut
port = int(os.environ.get("PORT", 8000))
ft.app(target=main, host="0.0.0.0", port=port, upload_dir="uploads")
