from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# "Hyperparameters"
w, h = 1024,512
first_iteration = True

def textureFromNumpy(tex_array):
	texture = glGenTextures(1)
	glPixelStorei(GL_UNPACK_ALIGNMENT,1)
	glBindTexture(GL_TEXTURE_2D, texture)
	print("bound texture")
	# Texture parameters are part of the texture object, so you need to
	# specify them only once for a given texture object.
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	print("set parameters")
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_array.shape[0], tex_array.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_array)
	print("set to image data")
	return texture

# Renders fullscreen quad
def fullscreenQuad():
	glBegin(GL_QUADS)
	glTexCoord2f(0, 0)
	glVertex2f(0, 0)
	glTexCoord2f(0, 1)
	glVertex2f(0, h)
	glTexCoord2f(1, 1)
	glVertex2f(w, h)
	glTexCoord2f(1, 0)
	glVertex2f(w, 0)
	glEnd()

def drawLoop():
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glViewport(0, 0, w, h)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	glOrtho(0.0, w, 0.0, h, -1.0, 1.0)
	glMatrixMode (GL_MODELVIEW)
	glLoadIdentity()

	glBindTexture(GL_TEXTURE_2D, texture)
	glEnable(GL_TEXTURE_2D)
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_TEXTURE_COORD_ARRAY)

	fullscreenQuad()

	# Hack to fix window interaction requirement.
	global first_iteration
	if (first_iteration):
		glutReshapeWindow(w, h)
		first_iteration = False

	glutSwapBuffers()


# Initialize GLU.
glutInit()
print("hey")
# Texture array.
print("fuck")
# We are rendering to RBGA.
glutInitDisplayMode(GLUT_RGBA)
# Set up the window. Resize in the draw loop on first time.
glutInitWindowSize(w-1, h-1)
glutInitWindowPosition(0, 0)
wind = glutCreateWindow("Render Texture")
# Set our draw loop to the display function.
glutDisplayFunc(drawLoop)
glutIdleFunc(drawLoop)
print("hi")

tex_array = np.full((w, h, 3), fill_value=255, dtype=np.uint8)
texture = textureFromNumpy(tex_array)

# Run the main loop.
glutMainLoop()
