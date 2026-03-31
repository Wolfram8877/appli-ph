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

print("Chargement des IA sur le serveur...")
# 1. On charge les DEUX cerveaux
modele_knn = joblib.load('modele_knn_ph.pkl')
modele_vectors = joblib.load('modele_vectors_ph.pkl') # <-- MODIFIE LE NOM SI BESOIN

def recadrer_image(chemin_image):
    image = Image.open(chemin_image).convert('RGB')
    
    # Version ultra-rapide avec NumPy
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
    
    # 2. On fait les deux prédictions
    ph_knn = round(float(modele_knn.predict(moyenne_rgb)[0]), 1)
    ph_vectors = round(float(modele_vectors.predict(moyenne_rgb)[0]), 1)
    
    # On renvoie les deux valeurs
    return ph_knn, ph_vectors

def main(page: ft.Page):
    page.title = "IA pH Analyzer"
    page.horizontal_alignment = "center"
    page.theme_mode = "light"
    page.scroll = "auto"

    titre = ft.Text("Testeur de pH par IA", size=30, weight="bold")
    
    # 3. On crée deux lignes de texte pour les résultats
    texte_resultat_knn = ft.Text("pH (Modèle KNN) : En attente...", size=18)
    texte_resultat_vectors = ft.Text("pH (Modèle Vecteurs) : En attente...", size=18)
    
    conteneur_image = ft.Container(width=300, height=300, border=ft.border.all(1, "grey"), border_radius=10)

    def on_upload(e: ft.FilePickerUploadEvent):
        if e.progress == 1.0: 
            chemin_fichier = os.path.join("uploads", e.file_name)
            
            with open(chemin_fichier, "rb") as image_file:
                code_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            conteneur_image.content = ft.Image(src_base64=code_image, width=300, height=300, fit="contain")
            texte_resultat_knn.value = "Analyse IA en cours..."
            texte_resultat_vectors.value = "" # On vide la 2ème ligne pendant l'analyse
            page.update()

            try:
                img = recadrer_image(chemin_fichier)
                
                # On récupère nos deux calculs
                ph_knn, ph_vectors = calculer_ph(img)
                
                # On met à jour l'affichage KNN
                texte_resultat_knn.value = f"pH ESTIMÉ (KNN) : {ph_knn}"
                texte_resultat_knn.color = "green" if 6 <= ph_knn <= 8 else "red"
                
                # On met à jour l'affichage Vecteurs
                texte_resultat_vectors.value = f"pH ESTIMÉ (Vecteurs) : {ph_vectors}"
                texte_resultat_vectors.color = "green" if 6 <= ph_vectors <= 8 else "red"
                
            except Exception as err:
                texte_resultat_knn.value = f"Erreur IA : {err}"
                texte_resultat_knn.color = "red"
            page.update()

    def on_result(e: ft.FilePickerResultEvent):
        if e.files:
            try:
                lien_original = page.get_upload_url(e.files[0].name, 60)
                from urllib.parse import urlparse
                parsed = urlparse(lien_original)
                lien_relatif = f"{parsed.path}?{parsed.query}"
                
                texte_resultat_knn.value = "Envoi de l'image..."
                texte_resultat_knn.color = "blue"
                texte_resultat_vectors.value = ""
                page.update()
                
                selecteur.upload([
                    ft.FilePickerUploadFile(e.files[0].name, upload_url=lien_relatif)
                ])
                
            except Exception as err:
                texte_resultat_knn.value = f"Erreur : {err}"
                texte_resultat_knn.color = "red"
                page.update()

    selecteur = ft.FilePicker()
    selecteur.on_result = on_result
    selecteur.on_upload = on_upload
    page.overlay.append(selecteur)

    bouton = ft.ElevatedButton("Prendre une photo", on_click=lambda _: selecteur.pick_files())

    # 4. Ne pas oublier d'ajouter nos deux textes à la page finale !
    page.add(titre, bouton, conteneur_image, texte_resultat_knn, texte_resultat_vectors)

port = int(os.environ.get("PORT", 8000))
ft.app(target=main, host="0.0.0.0", port=port, upload_dir="uploads")
