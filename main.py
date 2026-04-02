import flet as ft
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import joblib
import os
from urllib.parse import urlparse
import base64
import io

os.environ["FLET_SECRET_KEY"] = "super_cle_secrete_ph_123"

if not os.path.exists("uploads"):
    os.makedirs("uploads")

print("Chargement des IA sur le serveur...")
modele_knn = joblib.load('modele_knn_ph.pkl')
modele_vectors = joblib.load('modele_vectors_ph.pkl')
modele_RFR = joblib.load('modele_RFR_ph.pkl')
modele_DTR = joblib.load('modele_DTR_ph.pkl')

# --- Fonction d'optimisation---
def redimensionner_image(image_pil, max_size=2000):
    """
    Reduit l'image si elle est trop grande,
    tout en conservant ses proportions, et corrige l'orientation si besoin (EXIF).
    """
    # ImageOps.exif_transpose remet l'image a l'endroit (les telephones la tournent parfois)
    img = ImageOps.exif_transpose(image_pil) 
    img.thumbnail((max_size, max_size))
    return img

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

def recadrer_image(image_pil, centre_x, centre_y): 
    # Pour s'assurer qu'on ne sort pas des bords si on deplace trop le carre
    return image_pil.convert('RGB').crop((centre_x - 100, centre_y - 150, centre_x + 100, centre_y + 150))
    
def calculer_ph(image_decoupee):
    image_numpy = np.array(image_decoupee)
    moyenne_rgb = image_numpy.mean(axis=(0, 1)).reshape(1, -1)
    
    ph_knn = round(float(modele_knn.predict(moyenne_rgb)[0]), 1)
    ph_vectors = round(float(modele_vectors.predict(moyenne_rgb)[0]), 1)
    ph_RFR = round(float(modele_RFR.predict(moyenne_rgb)[0]), 1)
    ph_DTR = round(float(modele_DTR.predict(moyenne_rgb)[0]), 1)
    
    return ph_knn, ph_vectors, ph_RFR, ph_DTR

def main(page: ft.Page):
    page.title = "pH'ocus"
    page.horizontal_alignment = "center"
    page.theme_mode = "light"
    page.scroll = "auto"

    # --- OPTIMISATION : On stocke directement l'objet Image en RAM, plus le chemin
    etat_app = {"image_memoire": None, "cx": 0, "cy": 0}

    titre = ft.Image(
        src="icon.png", 
        width=300, 
        height=120,
        fit=ft.ImageFit.CONTAIN
    )

    texte_resultat_knn = ft.Text("pH Modele KNN : En attente...", size=18)
    texte_resultat_vectors = ft.Text("pH Modele Vecteurs : En attente...", size=18)
    texte_resultat_RFR = ft.Text("pH Modele RFR : En attente...", size=18)
    texte_resultat_DTR = ft.Text("pH Modele DTR : En attente...", size=18)
    
    conteneur_image = ft.Container(width=300, height=300, border=ft.border.all(1, "grey"), border_radius=10)

    def decaler_centre(dx, dy):
        # On verifie qu'une image est bien chargee en memoire
        if etat_app["image_memoire"] is None:
            return
            
        etat_app["cx"] += dx
        etat_app["cy"] += dy
        
        # --- OPTIMISATION : On utilise l'image deja en RAM
        original_image = etat_app["image_memoire"]
        cx, cy = etat_app["cx"], etat_app["cy"]
        
        original_image_editable = original_image.copy()
        draw = ImageDraw.Draw(original_image_editable)
        coordonnees_boite = (cx - 100, cy - 150, cx + 100, cy + 150)
        draw.rectangle(coordonnees_boite, outline="red", width=5)
        
        buffer_img = io.BytesIO()
        # On peut baisser un peu la qualite d'affichage (JPEG) pour accelerer le transfert Base64
        original_image_editable.save(buffer_img, format="JPEG", quality=85)
        code_image = base64.b64encode(buffer_img.getvalue()).decode('utf-8')
        conteneur_image.content.src_base64 = code_image
        
        try:
            img_crop = recadrer_image(original_image, cx, cy)
            ph_knn, ph_vectors, ph_RFR, ph_DTR = calculer_ph(img_crop)
            
            texte_resultat_knn.value = f"pH Modele KNN : {ph_knn}"
            texte_resultat_knn.color = "green" if 6 <= ph_knn <= 8 else "red"
            
            texte_resultat_vectors.value = f"pH Modele Vecteurs : {ph_vectors}"
            texte_resultat_vectors.color = "green" if 6 <= ph_vectors <= 8 else "red"

            texte_resultat_RFR.value = f"pH Modele RFR : {ph_RFR}"
            texte_resultat_RFR.color = "green" if 6 <= ph_RFR <= 8 else "red"

            texte_resultat_DTR.value = f"pH Modele DTR : {ph_DTR}"
            texte_resultat_DTR.color = "green" if 6 <= ph_DTR <= 8 else "red"
            
        except Exception as err:
            texte_resultat_knn.value = f"Erreur IA : {err}"
            texte_resultat_knn.color = "red"
            
        page.update()

   # Boutons
    btn_haut = ft.IconButton(icon=ft.Icons.ARROW_UPWARD, on_click=lambda _: decaler_centre(0, -50), visible=False)
    btn_bas = ft.IconButton(icon=ft.Icons.ARROW_DOWNWARD, on_click=lambda _: decaler_centre(0, 50), visible=False)
    btn_gauche = ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: decaler_centre(-50, 0), visible=False)
    btn_droite = ft.IconButton(icon=ft.Icons.ARROW_FORWARD, on_click=lambda _: decaler_centre(50, 0), visible=False)
    
    ligne_controles = ft.Row([btn_gauche, btn_haut, btn_bas, btn_droite], alignment=ft.MainAxisAlignment.CENTER)

    # --- GESTION DU CLAVIER ---
    def on_keyboard(e: ft.KeyboardEvent):
        # On verifie quelle touche a ete pressee et on decale le centre
        if e.key == "Arrow Up":
            decaler_centre(0, -50)
        elif e.key == "Arrow Down":
            decaler_centre(0, 50)
        elif e.key == "Arrow Left":
            decaler_centre(-50, 0)
        elif e.key == "Arrow Right":
            decaler_centre(50, 0)
            
    # Ecouteur d'evenements a la page
    page.on_keyboard_event = on_keyboard

    def on_upload(e: ft.FilePickerUploadEvent):
        if e.progress == 1.0: 
            chemin_fichier = os.path.join("uploads", e.file_name)
            
            # --- OPTIMISATION : On ouvre, on reduit la taille, et on stocke en RAM
            image_brute = Image.open(chemin_fichier)
            image_optimisee = redimensionner_image(image_brute)
            
            etat_app["image_memoire"] = image_optimisee
            etat_app["cx"], etat_app["cy"] = trouver_centre(image_optimisee)
            
            # On peut meme supprimer le fichier physique pour economiser l'espace serveur
            try:
                os.remove(chemin_fichier)
            except:
                pass

            btn_haut.visible = True
            btn_bas.visible = True
            btn_gauche.visible = True
            btn_droite.visible = True
            
            texte_resultat_vectors.value = "" 
            texte_resultat_RFR.value = ""
            texte_resultat_DTR.value = ""
            texte_resultat_knn.value = "Analyse en cours..."
            texte_resultat_knn.color = "blue"
            
            conteneur_image.content = ft.Image(src_base64="", width=300, height=300, fit="contain")
            page.update()
            
            decaler_centre(0, 0)

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
                texte_resultat_RFR.value = ""
                texte_resultat_DTR.value = ""
                
                btn_haut.visible = False
                btn_bas.visible = False
                btn_gauche.visible = False
                btn_droite.visible = False
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

    page.add(titre, bouton, conteneur_image, ligne_controles, texte_resultat_knn, texte_resultat_vectors, texte_resultat_RFR, texte_resultat_DTR)

port = int(os.environ.get("PORT", 8000))
ft.app(target=main, host="0.0.0.0", port=port, view=ft.AppView.WEB_BROWSER, upload_dir="uploads", assets_dir="assets")
