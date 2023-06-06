from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector

import numpy as np
from numpy import sin, cos
import cv2
import pandas as pd


IMU_DATA_LOC = "../IMUData.csv"

width = 500
height = 300

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

################################# PROBLEM 1 UTILS #################################
def gen_model(path="data/headset.obj"):
	model = Model(path)
	model.normalizeGeometry()
	return model
	

# These variables were used to test the near and far plane of
# the frustrum whilst keeping the general shape same
multiplier = 100
default_l_val = 1.875#width/(2*multiplier)
default_b_val = height/(2*multiplier)
default_n = -4

# from default values above, create new l, b values with near postion of near plane (for testing)
n=-7
l_val =np.tan(np.arctan(default_l_val/default_n))*n
b_val = np.tan(np.arctan(default_b_val/default_n))*n

def make_homogeneous(mat_3d):
	"""Converts a 3x3 matrix into homogenous form - 4x4"""
	return np.array([
			[*mat_3d[0], 0],
			[*mat_3d[1], 0],
			[*mat_3d[2], 0],
			[0,0,0,1]
		])
def get_canonical_transform_mat(n=n, l=-l_val, r=l_val, b=-b_val, t=b_val, f=-10):
	T_p = np.array([
		[n, 0, 0, 0],
		[0, n, 0, 0],
		[0, 0, n+f, -f*n],
		[0, 0, 1, 0],
		])

	T_st = np.array([
		[2/(r-l), 0, 0, -(r+l)/(r-l)],
		[0, (2/(t-b)), 0, -(t+b)/(t-b)],
		[0, 0, 2/(n-f), -(n+f)/(n-f)],
		[0, 0, 0, 1]
	])


	# The below could be used if the frustrum parameters were
	# selected such that r=-l and t=-b 
	# return np.array([
	# 	[1/r, 0, 0, 0],
	# 	[0, 1/t, 0, 0],
	# 	[0, 0, -2/(f-n), -(f+n)/(f-n)],
	# 	[0, 0, 0, 1]
	# ])
	return np.dot(T_st, T_p)

def get_viewport_transform_mat(m=500, n=300):
	T_vp = np.array([
		[m/2, 0, 0, (m-1)/2],
		[0, n/2, 0, (n-1)/2],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
		])
	return T_vp

def get_eye_transform_mat(orientation=None, translate_x=0, translate_y=0, translate_z=3):
	"""Function to return the eyetransformation matrix.
	Pass in IMU orientation matrix as the orientation parameter
	to move the camera to a particular orientation. Leave it
	as None to return the default camera orientation.

	Orientation is expected to be a 3x3 matrix
	"""
	e = np.array([translate_x,translate_y,translate_z])
	p = np.array([translate_x,translate_y,-1]) # [0,0,-1]
	v = p-e
	c_hat = v/np.linalg.norm(v)

	u_hat = np.array([0,1,0])

	z_hat = -c_hat
	x_hat = np.cross(u_hat, z_hat)
	y_hat = np.cross(z_hat, x_hat)
	if orientation is None:
		orientation = [
			[*x_hat],
			[*y_hat],
			[*z_hat],
		]
	else:
		orientation = np.array(orientation).T
	T_eye = np.matmul(
		make_homogeneous(orientation),
		np.array([
			[1,0,0, -e[0]],
			[0,1,0, -e[1]],
			[0,0,1, -e[2]],
			[0,0,0, 1]
			])
	)
	return T_eye

def project_vertex(projection_matrix, x, y, z, w=1):
	coords = np.array([[x], [y], [z], [w]])
	projection = np.matmul(projection_matrix, coords).flatten()
	pX = int(projection[0] / projection[3])
	pY = int(projection[1] / projection[3])
	pZ = projection[2] / projection[3]
	return pX, pY, pZ

can_trans_mat = get_canonical_transform_mat()
vp_trans_mat = get_viewport_transform_mat(width, height)




def get_rotation_mat(gamma, axis="x"):
	"""Function to generate a homogeneous rotation matrix.
	specificy axis as x, y or z to perform rotation by angle
	gamma (radians) in the corresponding axis.
	"""

	c = np.cos(gamma)
	s = np.sin(gamma)

	if axis == 'x':
		# rotate about x-axis
		R = np.array([[1, 0, 0, 0],
						[0, c, -s, 0],
						[0, s, c, 0],
						[0, 0, 0, 1]])
	elif axis == 'y':
		# rotate about y-axis
		R = np.array([[c, 0, s, 0],
						[0, 1, 0, 0],
						[-s, 0, c, 0],
						[0, 0, 0, 1]])
	else: #axis == 'z'
		# rotate about z-axis
		R = np.array([[c, -s, 0, 0],
						[s, c, 0, 0],
						[0, 0, 1, 0],
						[0, 0, 0, 1]])
	return R

def get_trans_mat(x, y, z):
	"""Function to generate a homogeneous translation matrix
	that translates in the x, y and z direction by the input
	paramters.
	"""
	return np.array([
		[1, 0, 0, x],
		[0, 1, 0, y],
		[0, 0, 1, z],
		[0, 0, 0, 1]
	])


def get_scale_mat(sf_x, sf_y, sf_z):
	"""Function to generate a homogeneous scaling matrix
	sf_x, sf_y, sf_z specify the scaling factor in the x, y
	and z directions respectively.
	"""
	return np.array([
		[sf_x, 0, 0, 0],
		[0, sf_y, 0, 0],
		[0, 0, sf_z, 0],
		[0, 0, 0, 1]
	])


def apply_trans_to_rot(rot_mat, x=0, y=0, z=0):
	"""Get a homogenous matrix that adds translation to input rotation matrix"""
	tran_rot_mat = rot_mat[:]
	tran_rot_mat[:, 3] = [x, y, z, 1]
	return tran_rot_mat
###################################################################################


def draw(model, rb_mat, image, zBuffer, eye_trans_mat, combined_trans_mat, viewport_trans_mat):
	# Calculate face normals
	faceNormals = {}
	for face in model.faces:
		p0, p1, p2 = [model.vertices[i] for i in face]
		faceNormal = (p2-p0).cross(p1-p0).normalize()
		homo_vertNorm = np.array([faceNormal.x, faceNormal.y, faceNormal.z])
		transformed_normal = np.matmul(
			np.linalg.inv(np.matmul(eye_trans_mat, rb_mat)[:3, :3]).T,
			homo_vertNorm
		)
		faceNormal = Vector(*transformed_normal)
		for i in face:
			if not i in faceNormals:
				faceNormals[i] = []

			faceNormals[i].append(faceNormal)

	# Calculate vertex normals
	vertexNormals = []
	for vertIndex in range(len(model.vertices)):
		vertNorm = getVertexNormal(vertIndex, faceNormals)
		vertexNormals.append(vertNorm)

	# Render the image iterating through faces
	for face in model.faces:
		p0, p1, p2 = [model.vertices[i] for i in face]
		n0, n1, n2 = [vertexNormals[i] for i in face]

		# Define the light direction
		lightDir = Vector(0, 0, -1)

		# Set to true if face should be culled
		cull = False

		# Transform vertices and calculate lighting intensity per vertex
		transformedPoints = []
		for p, n in zip([p0, p1, p2], [n0, n1, n2]):
			intensity = n * lightDir

			# Intensity < 0 means light is shining through the back of the face
			# In this case, don't draw the face at all ("back-face culling")
			if intensity < 0:
				cull = True
				break
			
			intensity = min(intensity,1)

			can_trans_res = np.matmul(combined_trans_mat, rb_mat)
			# proj_x, proj_y, proj_z = project_vertex(can_trans_res, p.x, p.y, p.z, 1)
			proj_x, proj_y, proj_z, proj_w = np.matmul(can_trans_res, [p.x, p.y, p.z, 1])
			proj_x, proj_y, proj_z = proj_x/proj_w, proj_y/proj_w, proj_z/proj_w 

			### Barrel Distortion - choose c_1 < 0
			proj_x, proj_y = apply_distortion(proj_x, proj_y, -0.2, 0.02)
			proj_x, proj_y, proj_z = project_vertex(viewport_trans_mat, proj_x, proj_y, proj_z, 1)


			transformedPoints.append(
				Point(
					int(proj_x),
					int(proj_y),
					proj_z,
					Color(intensity*255, intensity*255, intensity*255, 255)
					)
				)
		if not cull:
			Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw(image, zBuffer)


################################# PROBLEM 2 UTILS #################################

IMU_data = pd.read_csv(IMU_DATA_LOC)

# note there will be an insignificant difference between this and np.radians()
def deg_to_rad(x):
	return x*np.pi/180


def normalise(x, y, z):
    mag = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by 0
    if mag != 0:
        return x/mag, y/mag, z/mag
    return x, y, z

def euler_to_quat(euler_rads):
	"""Convert Euler (radians) to quaternions
	inspired by https://math.stackexchange.com/questions/2975109/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
	and https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
	roll (x), pitch (Y), yaw (z)
	"""
	roll, pitch, yaw = euler_rads
	cy = np.cos(yaw * 0.5)
	sy = np.sin(yaw * 0.5)
	cp = np.cos(pitch * 0.5)
	sp = np.sin(pitch * 0.5)
	cr = np.cos(roll * 0.5)
	sr = np.sin(roll * 0.5)

	# Calculate quaternion elements
	qw = cy * cp * cr + sy * sp * sr
	qx = cy * cp * sr - sy * sp * cr
	qy = sy * cp * sr + cy * sp * cr
	qz = sy * cp * cr - cy * sp * sr
	return (qw, qx, qy, qz)

def quaternion_to_euler(q):
	"""Inspired by https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
	and
	https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
	"""
	w, x, y, z = q

	sinr_cosp = 2 * (w * x + y * z)
	cosr_cosp = 1 - 2 * (x**2 + y**2)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	sinp = 2 * (w * y - z * x)
	sinp = 1 if sinp > 1 else sinp
	sinp = -1 if sinp < -1 else sinp
	pitch = np.arcsin(sinp)

	siny_cosp = 2 * (w * z + x * y)
	cosy_cosp = 1 - 2 * (y**2 + z**2)
	yaw = np.arctan2(siny_cosp, cosy_cosp)
	return (roll, pitch, yaw)

def quat_conj(quat):
	"""Caluclate the conjugate of a quaternion"""
	w, x, y, z = quat
	return w, -x, -y, -z

def quat_prod(quat_a, quat_b):
	"""Calculate the product of two quaternions"""
	w_a, x_a, y_a, z_a = quat_a
	w_b, x_b, y_b, z_b = quat_b

	w = w_a*w_b - x_a*x_b - y_a*y_b - z_a*z_b
	x = w_a*x_b + x_a*w_b + y_a*z_b - z_a*y_b
	y = w_a*y_b - x_a*z_b + y_a*w_b + z_a*x_b
	z = w_a*z_b + x_a*y_b - y_a*x_b + z_a*w_b
	return w, x, y, z

def quat_from_angle_axis(angle, axis):
	"""angle in radians. axis list in the form [x, y, z]"""
	return (np.cos(angle/2), axis[0]*np.sin(angle/2), axis[1]*np.sin(angle/2), axis[2]*np.sin(angle/2))

def angle_axis_from_quat(quat):
	qw, qx, qy, qz = quat
	s = np.sqrt(1-qw*qw)
	angle = 2 * np.arccos(qw)
	if abs(qw) != 1:
		x = qx / s
		y = qy / s
		z = qz / s
	else:
		x, y, z = 1, 0, 0
	return (angle, [x, y, z])
	
################################# PROBLEM 3 UTILS #################################
def dead_reckoning(q_prev_row, imu_row):
	"""Dead reckoning filter implementation using gyroscope readings and previos quaternion
	"""
	gyro_x, gyro_y, gyro_z = imu_row["gyroscope.X.rad"], imu_row["gyroscope.Y.rad"], imu_row["gyroscope.Z.rad"]
	prev_t = q_prev_row["time"]
	q_prev = q_prev_row["quaternion"]
	curr_t = imu_row["time"]

	mag = (gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)**0.5

	gyro_x /= mag
	gyro_y /= mag
	gyro_z /= mag

	delta_t = curr_t - prev_t
	delta_theta = mag*delta_t

	quat_i = quat_from_angle_axis(delta_theta, [gyro_x, gyro_y, gyro_z])
	q_curr = quat_prod(q_prev, quat_i)
	return q_curr


def tilt_correction(q_prev_row, imu_row, alpha=0.05):
	"""
	Function to include the accelerometer information for tilt correction and fuses
	the gyroscope info as well to create a complementary filter.
	q_prev_row: prev orientation
	"""
	dead_reckoning_orientation = dead_reckoning(q_prev_row, imu_row) # use previous fused orientation value
	acc_x, acc_y, acc_z = imu_row[" accelerometer.X"], imu_row[" accelerometer.Y"], imu_row[" accelerometer.Z"]
	# a^~ linear accelaration quaternion
	lin_acc_q = (0, acc_x, acc_y, acc_z)
	# a^hat = q * a^~ * q^-1 to bring back into global frame
	global_trans_helper = quat_prod(dead_reckoning_orientation, lin_acc_q)
	global_trans = quat_prod(global_trans_helper, quat_conj(dead_reckoning_orientation))
	# Orthogonal to XY plane (since z is up).
	# Orthogonal to (a^hat_x, a^hat_y, 0) is (-a^hat_y, a^hat_x, 0)
	tilt_axis = (-global_trans[2], global_trans[1], 0)
	v1 = (global_trans[1], global_trans[2], global_trans[3])
	up = (0, 0, 1)
	# angle btween global trans and up vector
	tilt_err_phi = np.arccos(np.dot(v1, up) / (np.linalg.norm(v1) * np.linalg.norm(up)))
	alpha = alpha
	angle = -alpha * tilt_err_phi
	tilt_quat = quat_from_angle_axis(angle, [tilt_axis[0], tilt_axis[1], tilt_axis[2]])
	compl_filter_k = quat_prod(tilt_quat, dead_reckoning_orientation)
	return compl_filter_k, tilt_err_phi

def quat_to_3d_rot(quat):
	"""quat: (w, x, y, z)
	from LaVelle book DOUBLE CHECK!!!"""
	w, x, y, z = quat
	rot = np.array([
		[2*(x**2+y**2)-1, 2*(y*z-x*w), 2*(y*w+x*z)],
		[2*(y*z+x*w), 2*(x**2+z**2)-1, 2*(z*w-x*y)],
		[2*(y*w-x*z), 2*(z*w+x*y), 2*(x**2+w**2)-1]
	])
	return rot

# Convert rotational rate into radians/sec
IMU_data["gyroscope.X.rad"] = IMU_data[" gyroscope.X"].apply(deg_to_rad)
IMU_data["gyroscope.Y.rad"] = IMU_data[" gyroscope.Y"].apply(deg_to_rad)
IMU_data["gyroscope.Z.rad"] = IMU_data[" gyroscope.Z"].apply(deg_to_rad)

# Normalise accelerometer and magnetometer values
for _, row in IMU_data.iterrows():
	row[" accelerometer.X"], row[" accelerometer.Y"], row[" accelerometer.Z"] = normalise(row[" accelerometer.X"], row[" accelerometer.Y"], row[" accelerometer.Z"])
	row[" magnetometer.X"], row[" magnetometer.Y"], row[" magnetometer.Z "] = normalise(row[" magnetometer.X"], row[" magnetometer.Y"], row[" magnetometer.Z "])



START_ORIENTATION = (1, 0, 0, 0)
orientation_quats = pd.DataFrame(columns=["time", "quaternion"])
orientation_quats["time"] = IMU_data["time"][:]
orientation_quats.at[0, "quaternion"] = START_ORIENTATION
for index in range(1, len(orientation_quats)):
	q_prev_row = orientation_quats.iloc[index - 1]
	imu_row = IMU_data.iloc[index]
	orientation_quats.at[index, 'quaternion'] = dead_reckoning(q_prev_row, imu_row)


orientation_acc_corr_quats = pd.DataFrame(columns=["time", "quaternion"])
orientation_acc_corr_quats["time"] = IMU_data["time"][:]
orientation_acc_corr_quats.at[0, "quaternion"] = START_ORIENTATION
for index in range(1, len(orientation_acc_corr_quats)):
	# curr_orientation = orientation_quats.at[index, "quaternion"]
	prev_orientation = orientation_acc_corr_quats.iloc[index-1]
	imu_row = IMU_data.iloc[index]

	_, tilt_error_before_tc = tilt_correction(orientation_quats.iloc[index - 1], imu_row)
	orientation_acc_corr_quats.at[index, 'quaternion'], tilt_error = tilt_correction(prev_orientation, imu_row, 0.04)


every_nth_frame = 1 # This can be altered to render every nth frame


################################# PROBLEM 4 UTILS #################################
def polar_from_cart(x, y):
	r = np.sqrt(x**2 + y**2)
	theta = np.arctan2(y, x)
	return r, theta

def cart_from_polar(r, theta):
	x = r * cos(theta)
	y = r * sin(theta)
	return x, y

def browns_distortion(radius, c_1, c_2):
	return radius + c_1*(radius**3) + c_2*(radius**5)


def approx_inverse_distortion(radius, c_1, c_2):
	return (c_1*radius**2 + c_2*radius**4 + ((c_1**2)*radius**4) + ((c_2**2)*radius**8 )+ (2*c_1*c_2*radius**6)) / (1 + (4*c_1*radius**2) + (6*c_2*radius**4))

def apply_distortion(x, y, c_1, c_2):
	r, theta = polar_from_cart(x, y)
	distorted_r = browns_distortion(r, c_1, c_2)
	distorted_x, distorted_y = cart_from_polar(distorted_r, theta)
	return distorted_x, distorted_y






################################# PROBLEM 5 UTILS #################################
def get_weight(mass):
	GRAVITY = 9.81
	return mass * GRAVITY

def get_drag(velocity):
	DRAG_COEFFICIENT = 0.8 # Arbitary number chosen from research of similar shaped objects
	AIR_DENSITY = 1.225 # kg/m^3
	REF_AREA = .187*.277 # approximaly 187mm×277mm w*l https://www.playstation.com/en-us/ps-vr/tech-specs/
	drag = (DRAG_COEFFICIENT * AIR_DENSITY * velocity**2 * REF_AREA)/2
	return drag

def get_new_velocity(weight, drag_force, mass, current_velocity):
    acceleration = (weight + drag_force) / mass # This is not weight - drag because we assume weight will be passed in as a negative -> up is posiitve
    new_velocity = current_velocity + acceleration
    return new_velocity

class Sphere:
	def __init__(self, position, radius):
		self.position = np.array(position) # [x,y,z]
		self.radius = radius

	def get_position(self):
		return self.position

	def set_position(self, position):
		self.position = position
	
class VR_object():
	def __init__(self, model, mass, position, velocity=np.array([0,0,0])):
		self.model = model
		self.mass = mass
		self.velocity = velocity
		self.x = position[0]
		self.y = position[1]
		self.z = position[2]
		self.sphere = Sphere(
			[self.x, self.y, self.z],
			1
			)
	
	def set_velocity(self, velocity):
		velocity = velocity
		self.velocity = np.array(velocity)
	
	def get_velocity(self):
		return self.velocity

	def get_mass(self):
		return self.mass

	def get_x(self):
		return self.x
	
	def get_y(self):
		return self.y
	
	def get_z(self):
		return self.z
	
	def set_x(self, x):
		self.x = x
		old_sphere_pos = self.sphere.get_position()
		self.sphere.set_position(np.array([x, old_sphere_pos[1], old_sphere_pos[2]]))

	def set_y(self, y):
		self.y = y
		old_sphere_pos = self.sphere.get_position()
		self.sphere.set_position(np.array([old_sphere_pos[0], y, old_sphere_pos[2]]))

	def set_z(self, z):
		self.z = z
		old_sphere_pos = self.sphere.get_position()
		self.sphere.set_position(np.array([old_sphere_pos[0], old_sphere_pos[1], z]))
	
	def update_pos_w_velocity(self, time):
		# time that has lapsed in seconds
		velocity = self.get_velocity()
		dist_moved = velocity * time # D = s * t
		scaled_dist_moved = dist_moved/5
		self.set_x(self.get_x() + scaled_dist_moved[0])
		self.set_y(max(FLOOR_Y, self.get_y() + scaled_dist_moved[1])) # Dont let it go through floor
		self.set_z(self.get_z() + scaled_dist_moved[2])


# Set position of the floor
FLOOR_Y = -2

def is_on_floor(obj):
	if obj.get_y() <= FLOOR_Y:
		return True
	return False




live_window = "bufferDisplay"
cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)

frame_arr = []


gamma = 0
model1 = gen_model()
model2 = gen_model()
model3 = gen_model()
# model4 = gen_model()
# model5 = gen_model()


vr_headset_1 = VR_object(model1, 0.7, [-3,5,-3], np.array([6, 0,0])) 
vr_headset_2 = VR_object(model2, 0.7, [3,5,-3], np.array([-6,0,0]))
vr_headset_3 = VR_object(model3, 0.7, [6,5,-3], np.array([-12,0,0]))
# vr_headset_4 = VR_object(model4, 0.7, [-2,6,-10], np.array([8,0,0]))
# vr_headset_5 = VR_object(model5, 0.7, [3,6,-18], np.array([-8,0,22]))


# All registered headset objects
vr_headset_arr = [vr_headset_1, vr_headset_2, vr_headset_3]
# vr_headset_arr = [vr_headset_1, vr_headset_2, vr_headset_3, vr_headset_4, vr_headset_5]

def has_collided(sphere1, sphere2):
	distance = np.sqrt((sphere1.position[0] - sphere2.position[0])**2 +
						(sphere1.position[1] - sphere2.position[1])**2 +
						(sphere1.position[2] - sphere2.position[2])**2)
	return distance < (sphere1.radius + sphere2.radius)

def handle_sphere_collision(obj1, obj2):
	# Inspired by elastic collision formulas presented in https://studiofreya.com/3d-math-and-physics/simple-sphere-sphere-collision-detection-and-collision-response/
	dist = obj2.sphere.position - obj1.sphere.position
	dist_norm = np.linalg.norm(dist)

	unit_dist_norm = dist / dist_norm

	unit_tangent = np.array([-unit_dist_norm[1], unit_dist_norm[0], 0])

	obj1_v_norm = np.dot(obj1.velocity, unit_dist_norm)
	velocity1_tangent = np.dot(obj1.velocity, unit_tangent)
	obj2_v_norm = np.dot(obj2.velocity, unit_dist_norm)
	velocity2_tangent = np.dot(obj2.velocity, unit_tangent)

	updated_obj1_v_norm = (obj1_v_norm * (obj1.mass - obj2.mass) + 2 * obj2.mass * obj2_v_norm) / (obj1.mass + obj2.mass)
	updated_obj2_v_norm = (obj2_v_norm * (obj2.mass - obj1.mass) + 2 * obj1.mass * obj1_v_norm) / (obj1.mass + obj2.mass)

	updated_obj1_v_norm_vector_vector = updated_obj1_v_norm * unit_dist_norm
	updated_obj1_v_tan_vector = velocity1_tangent * unit_tangent
	updated_obj2_v_norm_vector_vector = updated_obj2_v_norm * unit_dist_norm
	updated_obj2_v_tan_vector = velocity2_tangent * unit_tangent

	total_obj1_v = updated_obj1_v_norm_vector_vector + updated_obj1_v_tan_vector
	total_obj2_v = updated_obj2_v_norm_vector_vector + updated_obj2_v_tan_vector

	total_obj1_v, total_obj2_v = np.array(total_obj1_v), np.array(total_obj2_v)
	return total_obj1_v, total_obj2_v

def handle_on_floor(obj):
	'''Handles the condition where the object hasnt collided with another obj but is either
	on the floor or in free-fall
	'''
	if not is_on_floor(obj):
		obj = simulate_fall(obj)
	else:
		
		obj_curr_velocity = obj.get_velocity() 
		decayed_obj_curr_velocity = obj_curr_velocity/2 # GRADUALLY DECAY VELOCITY
		# Negate the y component of velocity
		obj.set_velocity(np.array([decayed_obj_curr_velocity[0], -decayed_obj_curr_velocity[1], decayed_obj_curr_velocity[2]]))
	return obj

def simulate_fall(obj):
	'''Simulate the fall of an object with the effects of gravity and drag'''
	obj_curr_velocity = obj.get_velocity()
	obj_curr_y_velocity = obj_curr_velocity[1]
	obj_drag = get_drag(obj_curr_y_velocity)
	if obj_curr_y_velocity > 0:
		# Change direction of drag if object moves upwards
		obj_drag = -obj_drag
	obj_mass = obj.get_mass()
	obj_weight = -get_weight(obj_mass)

	obj_new_y_velocity = get_new_velocity(obj_weight, obj_drag, obj_mass, obj_curr_y_velocity)
	obj.set_velocity(np.array([obj_curr_velocity[0], obj_new_y_velocity, obj_curr_velocity[2]]))
	return obj



frequency = 1/256

for index, row in orientation_acc_corr_quats.iloc[::every_nth_frame, :].iterrows():
	# Reset frame and zBuffer
	image = Image(width, height, Color(255, 255, 255, 255))
	zBuffer = [-float('inf')] * width * height	

	##### Update headset physics
	collided_indexes = set()

	for i in range(len(vr_headset_arr)-1):
		for j in range(i+1, len(vr_headset_arr)):
			if has_collided(vr_headset_arr[i].sphere, vr_headset_arr[j].sphere):
				print('collided')
				v1_velocity, v2_velocity = handle_sphere_collision(vr_headset_arr[i], vr_headset_arr[j])
				vr_headset_arr[i].set_velocity(v1_velocity)

				vr_headset_arr[j].set_velocity(v2_velocity)

				collided_indexes.add(i)
				collided_indexes.add(j)

	for i in range(len(vr_headset_arr)):
		if i not in collided_indexes:
			vr_headset_arr[i] = handle_on_floor(vr_headset_arr[i])
		vr_headset_arr[i].update_pos_w_velocity((1/256)*every_nth_frame)

	###### End update headset physics

	rb_mat1 = apply_trans_to_rot(
		get_rotation_mat(gamma, "y"),
		vr_headset_arr[0].get_x(),
		vr_headset_arr[0].get_y(),
		vr_headset_arr[0].get_z()-3
		)
	rb_mat2 = apply_trans_to_rot(
		get_rotation_mat(-gamma, "y"),
		vr_headset_arr[1].get_x(),
		vr_headset_arr[1].get_y(),
		vr_headset_arr[1].get_z()-3
		)
	rb_mat3 = apply_trans_to_rot(
		get_rotation_mat(-gamma*0.5, "y"),
		vr_headset_arr[2].get_x(),
		vr_headset_arr[2].get_y(),
		vr_headset_arr[2].get_z()-3
		)
	# rb_mat4 = apply_trans_to_rot(
	# 	get_rotation_mat(gamma*2, "y"),
	# 	vr_headset_arr[3].get_x(),
	# 	vr_headset_arr[3].get_y(),
	# 	vr_headset_arr[3].get_z()-3
	# 	)
	# rb_mat5 = apply_trans_to_rot(
	# 	get_rotation_mat(gamma, "y"),
	# 	vr_headset_arr[4].get_x(),
	# 	vr_headset_arr[4].get_y(),
	# 	vr_headset_arr[4].get_z()-3
	# 	)
	

	# ASSUMPTION: CAMERA ROTATES 
	# 1) ANTI-CLOCKWISE THEN CLOCKWISE,
	# 2) RIGHT THEN LEFT
	# 3) UP THEN DOWN
	imu_quat = row["quaternion"]

	new_quat = quat_conj(imu_quat)

	rotated_output = quat_prod((0,1,0,0), new_quat)
	orientation_rot_mat = quat_to_3d_rot(rotated_output)
	eye_trans_mat = get_eye_transform_mat(orientation_rot_mat)


	########################### FOR CAMERA ORIENTATIONS IN REPORT ###########################
	# test_rot_mat = get_rotation_mat(np.pi/4, "y")
	# eye_trans_mat = get_eye_transform_mat(test_rot_mat[:3,:3], 10, 5, 1)

	# test_rot_mat = get_rotation_mat(-np.pi/4, "y")
	# eye_trans_mat = get_eye_transform_mat(test_rot_mat[:3,:3], -8, 5, 1)
	
	# test_rot_mat = get_rotation_mat(np.pi/8, "z")
	# eye_trans_mat = get_eye_transform_mat(test_rot_mat[:3,:3], 1, 5, 5)

	# test_rot_mat = get_rotation_mat(-np.pi/8, "z")
	# eye_trans_mat = get_eye_transform_mat(test_rot_mat[:3,:3], 1, 5, 5)

	# test_rot_mat = get_rotation_mat(np.pi, "x")
	# eye_trans_mat = get_eye_transform_mat(test_rot_mat[:3,:3], 2, 5, 5)

	# eye_trans_mat = get_eye_transform_mat(None, 0, 10, 0)

	# eye_trans_mat = get_eye_transform_mat(None, 2, 5, 0)

	########################################################################################



	combined_trans_mat = np.matmul(can_trans_mat, eye_trans_mat)
	draw(vr_headset_arr[0].model, rb_mat1, image, zBuffer, eye_trans_mat, combined_trans_mat, vp_trans_mat)
	draw(vr_headset_arr[1].model, rb_mat2, image, zBuffer, eye_trans_mat, combined_trans_mat, vp_trans_mat)
	draw(vr_headset_arr[2].model, rb_mat3, image, zBuffer, eye_trans_mat, combined_trans_mat, vp_trans_mat)
	# draw(vr_headset_arr[3].model, rb_mat4, image, zBuffer, eye_trans_mat, combined_trans_mat, vp_trans_mat)
	# draw(vr_headset_arr[4].model, rb_mat5, image, zBuffer, eye_trans_mat, combined_trans_mat, vp_trans_mat)

	frame = cv2.imdecode(np.frombuffer(image.genPNG(), np.uint8), cv2.IMREAD_COLOR)
	frame_arr.append(frame)
	
	gamma += 0.01

	cv2.imshow(live_window, frame)
	# Since IMU readings are at a rate of 256Hz
	cv2.waitKey(int(frequency*1000))

# wait_time = int((1/256)*500)
# while True:
# 	for index, frame in enumerate(frame_arr):
# 		cv2.imshow(live_window, frame)
# 		cv2.waitKey(every_nth_frame*wait_time)
