class standard():
    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a standard car and has door:",self.doors)
        print("This is a standard car and is of size:", self.size)

class premium(standard):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a premium car and inherits from class:",premium.__bases__)
        print("This is a premium car and has door:", self.doors)
        print("This is a premium car and is of size:", self.size)


class sports(premium):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a sports car and inherits from class:",sports.__bases__)
        print("This is a sports car and has door:", self.doors)
        print("This is a sports car and is of size:", self.size)


class sedan(standard):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a sedan  and inherits from class:", sedan.__bases__)
        print("This is a sedan  and has door:", self.doors)
        print("This is a sedan  and is of size:", self.size)


class elite(premium):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a elite  and inherits from classes:", elite.__bases__)
        print("This is a elite  and has door:", self.doors)
        print("This is a elite  and is of size:", self.size)


class suv(premium):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is an suv  and inherits from class:", suv.__bases__)
        print("This is an suv  and has door:", self.doors)
        print("This is an suv  and is of size:", self.size)


class offRoad(suv, premium):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is an offroad  and inherits from class:", offRoad.__bases__)
        print("This is an offroad  and has door:", self.doors)
        print("This is an offroad  and is of size:", self.size)



class superCar(sports, premium):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a supercar  and inherits from class:", superCar.__bases__)
        print("This is a supercar  and has door:", self.doors)
        print("This is a supercar  and is of size:", self.size)


class luxury(premium,sedan):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is an standard  and inherits from class:", luxury.__bases__)
        print("This is an standard  and has door:", self.doors)
        print("This is an standard  and is of size:", self.size)


class compact(suv):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is an compact  and inherits from class:", compact.__bases__)
        print("This is an compact  and has door:", self.doors)
        print("This is an compact  and is of size:", self.size)


class executive(compact, suv):
    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is an executive  and inherits from class:", executive.__bases__)
        print("This is an executive  and has door:", self.doors)
        print("This is an executive  and is of size:", self.size)


class mini(elite, standard):

    def __init__(self, doors, size):
        self.doors = doors
        self.size = size

    def print_value(self):
        print("This is a mini  and inherits from class:", mini.__bases__)
        print("This is a mini  and has door:", self.doors)
        print("This is a mini  and is of size:", self.size)


obj1 = standard(doors = 4, size = 'mid_size' )
obj1.print_value()

obj2 = premium(doors = 6, size = 'mid_size' )
obj2.print_value()

obj3 = sports(doors = 2, size = 'small_size' )
obj3.print_value()

obj4 = sedan(doors = 6, size = 'mid_size' )
obj4.print_value()

obj5 = elite(doors = 2, size = 'small_size' )
obj5.print_value()

obj6 = suv(doors = 6, size = 'large_size' )
obj6.print_value()

obj7 = offRoad(doors = 4, size = 'small_size' )
obj7.print_value()

obj8 = superCar(doors = 2, size = 'small_size' )
obj8.print_value()

obj9 = luxury(doors = 8, size = 'small_size' )
obj9.print_value()

obj10 = compact(doors = 4, size = 'mid_size' )
obj10.print_value()

obj11 = executive(doors = 8, size = 'large_size' )
obj11.print_value()

obj12 = mini(doors = 2, size = 'small_size' )
obj12.print_value()

# OUTPUT:
# This is a standard car and has door: 4
# This is a standard car and is of size: mid_size
# This is a premium car and inherits from class: (<class '__main__.standard'>,)
# This is a premium car and has door: 6
# This is a premium car and is of size: mid_size
# This is a sports car and inherits from class: (<class '__main__.premium'>,)
# This is a sports car and has door: 2
# This is a sports car and is of size: small_size
# This is a sedan  and inherits from class: (<class '__main__.standard'>,)
# This is a sedan  and has door: 6
# This is a sedan  and is of size: mid_size
# This is a elite  and inherits from classes: (<class '__main__.premium'>,)
# This is a elite  and has door: 2
# This is a elite  and is of size: small_size
# This is an suv  and inherits from class: (<class '__main__.premium'>,)
# This is an suv  and has door: 6
# This is an suv  and is of size: large_size
# This is an offroad  and inherits from class: (<class '__main__.suv'>, <class '__main__.premium'>)
# This is an offroad  and has door: 4
# This is an offroad  and is of size: small_size
# This is a supercar  and inherits from class: (<class '__main__.sports'>, <class '__main__.premium'>)
# This is a supercar  and has door: 2
# This is a supercar  and is of size: small_size
# This is an standard  and inherits from class: (<class '__main__.premium'>, <class '__main__.sedan'>)
# This is an standard  and has door: 8
# This is an standard  and is of size: small_size
# This is an compact  and inherits from class: (<class '__main__.suv'>,)
# This is an compact  and has door: 4
# This is an compact  and is of size: mid_size
# This is an executive  and inherits from class: (<class '__main__.compact'>, <class '__main__.suv'>)
# This is an executive  and has door: 8
# This is an executive  and is of size: large_size
# This is a mini  and inherits from class: (<class '__main__.elite'>, <class '__main__.standard'>)
# This is a mini  and has door: 2
# This is a mini  and is of size: small_size
#
# Process finished with exit code 0



