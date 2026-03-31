import flet as ft
import numpy as np
from PIL import Image
import joblib
import os
from urllib.parse import urlparse
import base64

# Mot de passe Flet
os.environ["FLET_SECRET_KEY"] = "super_cle_secrete_ph_123"

# Création du dossier temporaire pour les images du téléphone
if not os.path.exists("uploads"):
    os.makedirs("uploads")

print("Chargement de l'IA sur le serveur...")
modele_ph = joblib.load('modele_knn_ph.pkl')

def recadrer_image(chemin_image):
    image = Image.open(chemin_image).convert('RGB')
    
    img_array = np.array(image, dtype=np.int16)
    diff = img_array.max(axis=-1) - img_array.min(axis=-1)
    y_coords, x_coords = np.where(diff > 100)
    
    if len(x_coords) > 0 and len(y_coords) > 0:
        centre_x = int((x_coords.min() + x_coords.max()) / 2)
        centre_y = int((y_coords.min() + y_coords.max()) / 2)
    else:
        l, h = image.size
        centre_x, centre_y = l // 2, h // 2
        
    return image.crop((centre_x - 100, centre_y - 150, centre_x + 100, centre_y + 150))

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

    # La fonction d'upload de Flet (inchangée)
    def on_upload(e):
        if e.progress == 1.0: 
            chemin_fichier = os.path.join("uploads", e.file_name)
            
            with open(chemin_fichier, "rb") as image_file:
                code_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            conteneur_image.content = ft.Image(src_base64=code_image, width=300, height=300, fit="contain")
            texte_resultat.value = "Analyse IA en cours..."
            page.update()

            try:
                img = recadrer_image(chemin_fichier)
                ph = calculer_ph(img)
                texte_resultat.value = f"pH ESTIMÉ : {ph}"
                texte_resultat.color = "green" if 6 <= ph <= 8 else "red"
            except Exception as err:
                texte_resultat.value = f"Erreur IA : {err}"
                texte_resultat.color = "red"
            page.update()

    selecteur = ft.FilePicker()
    selecteur.on_upload = on_upload
    page.overlay.append(selecteur)

    # LA NOUVELLE MAGIE FLET : Tout se passe directement ici !
    async def au_clic_bouton(e):
        # 1. On ouvre l'appareil photo et on attend le résultat (plus de 'on_result' !)
        files = await selecteur.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE)
        
        # 2. Si l'utilisateur a bien pris une photo (et n'a pas annulé)
        if files:
            try:
                fichier = files[0]
                lien_original = page.get_upload_url(fichier.name, 60)
                from urllib.parse import urlparse
                parsed = urlparse(lien_original)
                lien_relatif = f"{parsed.path}?{parsed.query}"
                
                texte_resultat.value = "Envoi de l'image..."
                texte_resultat.color = "blue"
                page.update()
                
                # 3. On lance l'envoi au serveur
                await selecteur.upload([
                    ft.FilePickerUploadFile(fichier.name, upload_url=lien_relatif)
                ])
                
            except Exception as err:
                texte_resultat.value = f"Erreur : {err}"
                texte_resultat.color = "red"
                page.update()

    bouton = ft.ElevatedButton("Prendre une photo", on_click=au_clic_bouton)

    page.add(titre, bouton, conteneur_image, texte_resultat)

port = int(os.environ.get("PORT", 8000))
ft.app(target=main, host="0.0.0.0", port=port, upload_dir="uploads")
