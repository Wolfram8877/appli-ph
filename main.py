import flet as ft
import numpy as np
from PIL import Image, ImageDraw  
import joblib
import os
from urllib.parse import urlparse
import base64
import io  

# Mot de passe Flet
os.environ["FLET_SECRET_KEY"] = "super_cle_secrete_ph_123"

# Création du dossier temporaire pour les images du téléphone
if not os.path.exists("uploads"):
    os.makedirs("uploads")

print("Chargement des IA sur le serveur...")
# On charge les 4 modèles d'IA entrainés
modele_knn = joblib.load('modele_knn_ph.pkl')
modele_vectors = joblib.load('modele_vectors_ph.pkl')
modele_RFR = joblib.load('modele_RFR_ph.pkl')
modele_DTR = joblib.load('modele_DTR_ph.pkl')

# --- FONCTION POUR TROUVER LE CENTRE ---
def trouver_centre(image_pil):
    img_array = np.array(image_pil.convert('RGB'), dtype=np.int16)
    diff = img_array.max(axis=-1) - img_array.min(axis=-1)
    y_coords, x_coords = np.where(diff > 100)
    
    if len(x_coords) > 0 and len(y_coords) > 0:
        centre_x = int((x_coords.min() + x_coords.max()) / 2)
        centre_y = int((y_coords.min() + y_coords.max()) / 2)
    else:
        l, h = image_pil.size
        centre_x, centre_y = l // 2, h // 2
    return centre_x, centre_y

# Cropping de l'image a centre (200x300 pixels)
def recadrer_image(image_pil, centre_x, centre_y): 
    return image_pil.convert('RGB').crop((centre_x - 100, centre_y - 150, centre_x + 100, centre_y + 150))

def calculer_ph(image_decoupee):
    image_numpy = np.array(image_decoupee)
    moyenne_rgb = image_numpy.mean(axis=(0, 1)).reshape(1, -1)
    
    # On fait les QUATRE prédictions avec les bons modèles
    ph_knn = round(float(modele_knn.predict(moyenne_rgb)[0]), 1)
    ph_vectors = round(float(modele_vectors.predict(moyenne_rgb)[0]), 1)
    ph_RFR = round(float(modele_RFR.predict(moyenne_rgb)[0]), 1)
    ph_DTR = round(float(modele_DTR.predict(moyenne_rgb)[0]), 1)
    
    # On retourne les 4 résultats
    return ph_knn, ph_vectors, ph_RFR, ph_DTR

def main(page: ft.Page):
    page.title = "pH Analyzer"
    page.horizontal_alignment = "center"
    page.theme_mode = "light"
    page.scroll = "auto"

    titre = ft.Text("Test de pH", size=30, weight="bold")
    
    # On crée 4 lignes de texte pour les résultats
    texte_resultat_knn = ft.Text("pH Modèle KNN : En attente...", size=18)
    texte_resultat_vectors = ft.Text("pH Modèle Vecteurs : En attente...", size=18)
    texte_resultat_RFR = ft.Text("pH Modèle RFR : En attente...", size=18)
    texte_resultat_DTR = ft.Text("pH Modèle DTR : En attente...", size=18)
    
    conteneur_image = ft.Container(width=300, height=300, border=ft.border.all(1, "grey"), border_radius=10)

    def on_upload(e: ft.FilePickerUploadEvent):
        if e.progress == 1.0: 
            chemin_fichier = os.path.join("uploads", e.file_name)
            
            # MODIFICATION POUR LE RECTANGLE ---
            original_image = Image.open(chemin_fichier)
            cx, cy = trouver_centre(original_image)
            
            original_image_editable = original_image.copy()
            draw = ImageDraw.Draw(original_image_editable)
            
            coordonnees_boite = (cx - 100, cy - 150, cx + 100, cy + 150)
            draw.rectangle(coordonnees_boite, outline="red", width=5)
            
            buffer_img = io.BytesIO()
            original_image_editable.save(buffer_img, format="PNG")
            code_image = base64.b64encode(buffer_img.getvalue()).decode('utf-8')
            
            conteneur_image.content = ft.Image(src_base64=code_image, width=300, height=300, fit="contain")
            texte_resultat_knn.value = "Analyse en cours..."
            
            # On vide toutes les autres lignes
            texte_resultat_vectors.value = "" 
            texte_resultat_RFR.value = ""
            texte_resultat_DTR.value = ""
            page.update()

            try:
                img = recadrer_image(original_image, cx, cy)
                
                # On récupère nos 4 calculs
                ph_knn, ph_vectors, ph_RFR, ph_DTR = calculer_ph(img)
                
                # Mise à jour KNN
                texte_resultat_knn.value = f"pH Modèle KNN : {ph_knn}"
                texte_resultat_knn.color = "green" if 6 <= ph_knn <= 8 else "red"
                
                # Mise à jour Vecteurs
                texte_resultat_vectors.value = f"pH Modèle Vecteurs : {ph_vectors}"
                texte_resultat_vectors.color = "green" if 6 <= ph_vectors <= 8 else "red"

                # Mise à jour RFR
                texte_resultat_RFR.value = f"pH Modèle RFR : {ph_RFR}"
                texte_resultat_RFR.color = "green" if 6 <= ph_RFR <= 8 else "red"

                # Mise à jour DTR
                texte_resultat_DTR.value = f"pH Modèle DTR : {ph_DTR}"
                texte_resultat_DTR.color = "green" if 6 <= ph_DTR <= 8 else "red"
                
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
                
                # On efface les autres textes pendant le chargement
                texte_resultat_vectors.value = ""
                texte_resultat_RFR.value = ""
                texte_resultat_DTR.value = ""
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

    # ON AJOUTE LES 4 TEXTES A LA PAGE ICI
    page.add(titre, bouton, conteneur_image, texte_resultat_knn, texte_resultat_vectors, texte_resultat_RFR, texte_resultat_DTR)

# LA LIGNE CLÉ POUR LE CLOUD RENDER :
port = int(os.environ.get("PORT", 8000))
ft.app(target=main, host="0.0.0.0", port=port, upload_dir="uploads")
