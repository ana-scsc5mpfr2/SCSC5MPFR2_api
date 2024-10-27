from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
import traceback

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
MAX_SIZE = 2024
V = 15
N = 2
U = 12
M = U - N

def validate_image(image):
    """Validate image size and format"""
    if image.size[0] > MAX_SIZE or image.size[1] > MAX_SIZE:
        raise ValueError(f"Image dimensions must not exceed {MAX_SIZE}x{MAX_SIZE} pixels")
    
    allowed_formats = {'PNG', 'JPEG', 'JPG'}
    if image.format.upper() not in allowed_formats:
        raise ValueError(f"Invalid image format. Allowed formats: {', '.join(allowed_formats)}")

def enhance_image(image_array):
    """Enhance image quality"""
    try:
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((l,a,b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image_array

def encontrar_colores_dominantes(imagen_array, num_colores=3):
    """Encuentra los colores dominantes usando K-means clustering"""
    pixels = imagen_array.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, num_colores, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    return centers

def generar_analisis_radial(imagen_array):
    """Genera análisis radial de la imagen"""
    altura, ancho = imagen_array.shape[:2]
    centro = (ancho // 2, altura // 2)
    radio = min(ancho, altura) // 2
    
    # Crear máscara circular
    mascara = np.zeros((altura, ancho), dtype=np.uint8)
    cv2.circle(mascara, centro, radio, 255, -1)
    
    # Aplicar máscara a la imagen
    imagen_procesada = imagen_array.copy()
    imagen_procesada[mascara == 0] = [0, 0, 0]
    
    # Generar líneas radiales y resultados
    resultados = []
    for angulo in range(360):
        rad = np.radians(angulo)
        x = int(centro[0] + radio * np.cos(rad))
        y = int(centro[1] - radio * np.sin(rad))
        
        # Dibujar línea
        cv2.line(imagen_procesada, centro, (x, y), (0, 255, 0), 1)
        
        # Calcular valor
        distancia = np.sqrt((x - centro[0])**2 + (y - centro[1])**2)
        valor = -V + (distancia / radio) * V
        
        resultados.append({
            "angle": float(angulo),
            "value": float(valor)
        })
    
    return imagen_procesada, resultados

def process_image(image_data):
    """Main image processing function"""
    try:
        # Convert base64 to image
        image_data = image_data.split(';base64,')[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Validate image
        validate_image(image)
        
        # Convert to numpy array
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Enhance image
        enhanced_image = enhance_image(image_array)
        
        # Find dominant colors
        colores_dominantes = encontrar_colores_dominantes(enhanced_image)
        
        # Generate radial analysis
        processed_image, vector_results = generar_analisis_radial(enhanced_image)
        
        # Convert processed image to base64
        success, buffer = cv2.imencode('.png', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Failed to encode processed image")
        
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'processed_image': f'data:image/png;base64,{processed_base64}',
            'vector_results': vector_results,
            'dominant_colors': colores_dominantes.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/process-image', methods=['POST'])
def handle_image_processing():
    logger.debug("Hola Soy Dora!")
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        result = process_image(data['image'])
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)