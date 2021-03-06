from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class TexRenderer():

	def __init__(self, w, h):
		# Open GL Initialization.
		glutInit()
		# We are rendering to RBGA.
		glutInitDisplayMode(GLUT_RGBA)
		# Set up the window. Resize in the draw loop on first time.
		glutInitWindowSize(w-1, h-1)
		glutInitWindowPosition(0, 0)
		wind = glutCreateWindow("Render Texture")

		# Set our draw loop to the display function.
		glutDisplayFunc(self.drawLoop)
		glutIdleFunc(self.drawLoop)

		# Member variables.
		self.w = w
		self.h = h
		self.tex_array = np.full((w, h, 3), fill_value=255, dtype=np.uint8)
		self.texture = self.textureFromNumpy(self.tex_array) #?
		self.first_iteration = True

	def textureFromNumpy(self, tex_array):
		texture = glGenTextures(1)
		glPixelStorei(GL_UNPACK_ALIGNMENT,1)
		glBindTexture(GL_TEXTURE_2D, texture)
		# Texture parameters are part of the texture object, so you need to
		# specify them only once for a given texture object.
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_array.shape[0], tex_array.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_array)
		return texture

	def drawLoop(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glViewport(0, 0, self.w, self.h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0.0, self.w, 0.0, self.h, -1.0, 1.0)
		glMatrixMode (GL_MODELVIEW)
		glLoadIdentity()


		self.updateTexture()

		glBindTexture(GL_TEXTURE_2D, self.texture)
		glEnable(GL_TEXTURE_2D)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)

		self.fullscreenQuad()

		# Hack to fix window interaction requirement.
		if (self.first_iteration):
			glutReshapeWindow(self.w, self.h)
			self.first_iteration = False

		glutSwapBuffers()

	def updateTexture():
		self.tex_array += 1
		self.tex_array %= 255
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.tex_array.shape[0],
			self.tex_array.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE,
			self.tex_array)

	# Renders fullscreen quad
	def fullscreenQuad(self):
		glBegin(GL_QUADS)
		glTexCoord2f(0, 0)
		glVertex2f(0, 0)
		glTexCoord2f(0, 1)
		glVertex2f(0, self.h)
		glTexCoord2f(1, 1)
		glVertex2f(self.w, self.h)
		glTexCoord2f(1, 0)
		glVertex2f(self.w, 0)
		glEnd()

	def run(self):
		glutMainLoop()


def main():
	tex_renderer = TexRenderer(1024, 512)
	tex_renderer.run()

main()
