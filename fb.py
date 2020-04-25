import pygame
import neat
import time
import os
import random
import pdb
pygame.font.init()

WIDTH = 250
HEIGHT = 400

Bird_img = [(pygame.image.load(os.path.join("assets","bird1.png"))),(pygame.image.load(os.path.join("assets","bird2.png"))),(pygame.image.load(os.path.join("assets","bird3.png"))),(pygame.image.load(os.path.join("assets","bird2.png")))]
Pipe_img = pygame.image.load(os.path.join("assets","pipe.png"))
Base_img = pygame.image.load(os.path.join("assets","base.png"))
Bg_img = pygame.image.load(os.path.join("assets","bg.png"))
Font = pygame.font.SysFont("Calibri",25)

class Bird:
	Max_rot = 25
	Max_vel = 12.5
	Animation_time = 10
	Animation_epoch = 40
	rot_v = 10
	Imgs = Bird_img

	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.h = y
		self.tilt = 0
		self.vel = 0
		self.tick_c = 0
		self.img_c = 0
		self.img = self.Imgs[0]

	def jump(self):
		self.vel = -6.5
		self.tick_c = 0
		self.h = self.y

	def move(self):
		self.tick_c += 1
		d = self.vel*(self.tick_c/2) + 1.5*(self.tick_c/2)**2
		if d > 8:
			d = 8
		elif d < 0:
			d -= 1
		self.y += d
		if d < 0 or self.y < self.h + 25:
				self.tilt = self.Max_rot
		else:
			if self.tilt > -90:
				self.tilt -= self.rot_v

	def draw(self,win):
		self.img_c += 1
		self.img_c %= self.Animation_epoch
		self.img = self.Imgs[int(self.img_c/self.Animation_time)]
		if self.tilt <= -80:
			self.img = self.Imgs[1]
			self.img_c = self.Animation_time*2
		rot = pygame.transform.rotate(self.img,self.tilt)
		nr = rot.get_rect(center = self.img.get_rect(topleft = (int(self.x),int(self.y))).center)
		win.blit(rot,nr.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.img)

class Pipe:
	def __init__(self,x):
		self.x = x
		self.h = 0
		self.gap = 100
		self.vel = 2.5
		self.top = 0
		self.bottom = 0
		self.pipeup = pygame.transform.flip(Pipe_img,False,True)
		self.pipedown = Pipe_img
		#self.passed = False
		self.set_height()

	def set_height(self):
		self.h = random.randrange(25,225)
		self.top = self.h - self.pipeup.get_height()
		self.bottom = self.h + self.gap

	def move(self):
		self.x -= self.vel

	def draw(self,win):
		win.blit(self.pipeup,(int(self.x),int(self.top)))
		win.blit(self.pipedown,(int(self.x),int(self.bottom)))

	def collide(self,bird):
		brm = bird.get_mask()
		tm = pygame.mask.from_surface(self.pipeup)
		bm = pygame.mask.from_surface(self.pipedown)
		to = (int(self.x - bird.x),int(self.top - int(round(bird.y))))
		bo = (int(self.x - bird.x),int(self.bottom - int(round(bird.y))))
		tc = brm.overlap(tm,to)
		bc = brm.overlap(bm,bo)
		if tc or bc:
			return True
		return False

class Base:
    vel = 2.5
    width = Base_img.get_width()
    img = Base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        if self.x1 + self.width< 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.img, (int(self.x1), int(self.y)))
        win.blit(self.img, (int(self.x2), int(self.y)))

def draw_window(win,birds,pipe,base,generation):
	win.blit(Bg_img,(0,0))
	for bird in birds:
		bird.draw(win)
	text = Font.render("Generation : "+str(generation),1,(255,255,255))
	pipe.draw(win)
	base.draw(win)
	win.blit(text,(WIDTH - 5 - text.get_width(),5))
	pygame.display.update()

def main(genomes, config):
	global gen
	nets = []
	birds = []
	ge = []
	for genome_id, genome in genomes:
		genome.fitness = 0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		birds.append(Bird(75,150))
		ge.append(genome)
	base = Base(365)
	pipe = Pipe(300)
	clock = pygame.time.Clock()
	WIN = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Flappy Bird")
	run = True
	while run:
		if len(birds) == 0:
			run = False
		clock.tick(60)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
				break
		base.move()
		pipe.move()
		for x, bird in enumerate(birds):
			bird.move()
			ge[birds.index(bird)].fitness += 0.1
			output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipe.h), abs(bird.y - pipe.bottom)))
			if output[0] > 0.5:
				bird.jump()
			if pipe.x < bird.x:
				ge[birds.index(bird)].fitness += 1
			elif pipe.collide(bird):
				ge[birds.index(bird)].fitness -= 2
				ge.pop(birds.index(bird))
				birds.pop(birds.index(bird))
			elif bird.y + bird.img.get_height() >= 365 or bird.y < 0:
				ge[birds.index(bird)].fitness -= 2
				ge.pop(birds.index(bird))
				birds.pop(birds.index(bird))
		if pipe.x + pipe.pipeup.get_width() < 0:
			pipe = Pipe(300)
		draw_window(WIN,birds,pipe,base,gen)
	gen += 1

def set_configurations():
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config.txt')
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	winner = p.run(main, 50)
	print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
	gen = 0
	set_configurations()