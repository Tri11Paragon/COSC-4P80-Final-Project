{ pkgs ? (import <nixpkgs> { 
    config.allowUnfree = true;
    config.segger-jlink.acceptLicense = true; 
}), customPkgs ? (import /home/brett/my-nixpkgs {
	config.allowUnfree = true;
    config.segger-jlink.acceptLicense = true;
}), ... }:
pkgs.mkShell
{
	buildInputs = with pkgs; [
		cmake 
		gcc
		clang
		emscripten
		ninja
		customPkgs.jetbrains.clion
		#clion = import ~/my-nixpkgs/pkgs/applications/editors/jetbrains {};
		renderdoc
		valgrind
	];
	propagatedBuildInputs = with pkgs; [
		xorg.libX11 
		xorg.libX11.dev
		xorg.libXcursor
		xorg.libXcursor.dev
		xorg.libXext 
		xorg.libXext.dev
		xorg.libXinerama
		xorg.libXinerama.dev 
		xorg.libXrandr 
		xorg.libXrandr.dev
		xorg.libXrender
		xorg.libXrender.dev
		xorg.libxcb
		xorg.libxcb.dev
		xorg.libXi
		xorg.libXi.dev
		harfbuzz
		harfbuzz.dev
		zlib
		zlib.dev
		bzip2
		bzip2.dev
		pngpp
		brotli
		brotli.dev
		pulseaudio.dev
		git
		libGL
		libGL.dev
		glfw
		openblas
		openblas.dev
	];
	LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib";
}
