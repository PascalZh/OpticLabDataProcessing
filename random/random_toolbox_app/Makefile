UI_SRC=src/generated/random_toolbox_main_window.py

all: $(UI_SRC)
	cp ../random_toolbox.py src/generated/random_toolbox.py

$(UI_SRC): src/generated/%.py : src/%.ui
	pyuic5 -o $@ $<
